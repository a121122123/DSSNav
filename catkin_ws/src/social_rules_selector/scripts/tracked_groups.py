#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Software License Agreement (BSD License)
#
#  Copyright (c) 2014-2015, Timm Linder, Social Robotics Lab, University of Freiburg
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#  * Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
根據 social relation 進行分群 (修改版)

此程式訂閱 topic "trk3d_result" (msg: ultralytics_ros/Trk3DArray)
以及 social relation 訊息 (SocialRelations)，
根據 social relation 的強度建立每對 track 間的「距離矩陣」，
再利用階層式分群 (single-linkage) 進行分群，
最後發佈群組分群結果 (TrackedGroups)。

作者：請自行修改
"""

import os, sys, math, time
from collections import deque
from multiprocessing import Lock

import rospy, tf, message_filters
import geometry_msgs.msg
import numpy, scipy, scipy.spatial.distance, scipy.cluster.hierarchy

# 原始群組訊息格式 (假設保持不變)
# from spencer_tracking_msgs.msg import TrackedGroups, TrackedGroup
# from spencer_social_relation_msgs.msg import SocialRelations, SocialRelation
# 新的群組訊息格式 (使用你提供的 msg 結構)
from social_rules_selector.msg import TrackedGroups, TrackedGroup
from social_rules_selector.msg import SocialRelations, SocialRelation
# 新的追蹤資料格式 (使用你提供的 msg 結構)
from ultralytics_ros.msg import Trk3DArray, Trk3D
from visualization_msgs.msg import Marker, MarkerArray


last_group_ids = set()
#########################################################################
# 回呼：收到新資料時根據 social relation 與追蹤資訊分群
#########################################################################
def newDataAvailable(trk3dArray, socialRelations):
    # rospy.loginfo("SocialRelations has %d elements", len(socialRelations.elements))
    
    # 使用 trk3dArray.trks_list 當作追蹤結果
    trackCount = len(trk3dArray.trks_list)

    # 建立從 tracked_id 到 0-base index 的映射
    trackIdToIndex = dict()
    trackIndex = 0
    for trk in trk3dArray.trks_list:
        trackIdToIndex[trk.tracked_id] = trackIndex
        trackIndex += 1

    # 建立一個大小為 (trackCount x trackCount) 的矩陣，初始值為 1.0
    # 我們以「1.0 - relation strength」作為距離，所以關係強度越大距離越小。
    socialRelationsMatrix = numpy.ones((trackCount, trackCount))
    # 針對每一對 social relation，更新矩陣中的對應數值
    for socialRelation in socialRelations.elements:
        try:
            index1 = trackIdToIndex[socialRelation.track1_id]
            index2 = trackIdToIndex[socialRelation.track2_id]
        except KeyError as e:
            rospy.logwarn("Key error while looking up tracks %d and %d!" %
                          (socialRelation.track1_id, socialRelation.track2_id))
            continue

        # 如果有重複關係，取 (1 - strength) 最小值
        socialRelationsMatrix[index1, index2] = min(socialRelationsMatrix[index1, index2],
                                                    1.0 - socialRelation.strength)
        socialRelationsMatrix[index2, index1] = min(socialRelationsMatrix[index2, index1],
                                                    1.0 - socialRelation.strength)

    # 對角線強制設定為0
    for i in range(trackCount):
        socialRelationsMatrix[i, i] = 0.0

    # 把完整矩陣轉換為 condensed form，方便後續分群
    trackDistances = scipy.spatial.distance.squareform(socialRelationsMatrix, force='tovector')

    # 以參數 relation_threshold 當作分群閥值 (關係強度超過此值視同同群)
    relationThreshold = rospy.get_param('~relation_threshold', 0.5)
    groupIndices = cluster(trackCount, trackDistances, relationThreshold)  # 回傳每一 track 的群組編號

    # 根據 groupIndices 產生組，key 為群組編號，value 為 track ID 的 list
    groups = createGroups(groupIndices, trk3dArray.trks_list)

    # 依據目前分群結果與歷史分群結果對應，保持群組 ID 穩定
    trackedGroups = trackGroups(groups, trk3dArray)

    # 發佈分群結果
    publishGroups(trackedGroups, trk3dArray)
    # 發佈群組視覺化結果
    publishGroupMarkers(trackedGroups, trk3dArray)


#########################################################################
# 以 single-linkage clustering 產生群組編號
#########################################################################
def cluster(trackCount, trackDistances, threshold):
    # 如果沒有足夠的 track，直接處理
    if trackCount == 0:
        return []
    elif trackCount == 1:
        return [0]
    linkage = scipy.cluster.hierarchy.linkage(trackDistances, method='single')
    groupIndices = scipy.cluster.hierarchy.fcluster(linkage, threshold, criterion='distance')
    return groupIndices


#########################################################################
# 根據群組編號與 tracks 資料產生群組 (dict: key 為群組編號, value 為 track ID list)
#########################################################################
def createGroups(groupIndices, tracks):
    groups = dict()
    trackIndex = 0
    for groupIndex in groupIndices:
        if groupIndex not in groups:
            groups[groupIndex] = []
        trackId = tracks[trackIndex].tracked_id
        groups[groupIndex].append(trackId)
        trackIndex += 1
    return groups


#########################################################################
# 以下部份用以記錄群組對應 (保持群組 ID 穩定)
#########################################################################
class GroupIdAssignment(object):
    def __init__(self, trackIds, groupId, createdAt):
        self.trackIds = set(trackIds)
        self.groupId = groupId
        self.createdAt = createdAt

    def __str__(self):
        return "%s = %d" % (str(list(self.trackIds)), self.groupId)

class GroupIdRemapping(object):
    def __init__(self, groupId, publishedGroupId):
        self.originalGroupId = groupId
        self.publishedGroupId = publishedGroupId

def remapGroupId(groupId):
    publishedGroupId = None
    for groupIdRemapping in trackGroups.groupIdRemapping:
        if groupIdRemapping.originalGroupId == groupId:
            publishedGroupId = groupIdRemapping.publishedGroupId
            break
    if publishedGroupId is None:
        trackGroups.largestPublishedGroupId += 1
        publishedGroupId = trackGroups.largestPublishedGroupId

    trackGroups.groupIdRemapping.append(GroupIdRemapping(groupId, publishedGroupId))
    return publishedGroupId


#########################################################################
# 將目前分群結果與歷史資訊比對，分配穩定的群組 ID
#########################################################################
def trackGroups(groups, trk3dArray):
    currentTime = trk3dArray.header.stamp
    publishSinglePersonGroups = rospy.get_param('~publish_single_person_groups', False)
    trackedGroups = []
    assignedGroupIds = []

    # 為了保證每次產生的群組順序固定，依每組中最小 track_id 排序
    sortedGroups = sorted(groups.items(), key=lambda item: sorted(item[1])[0])

    # 建立由 tracked_id 至位置與速度的字典
    trackPositionsById = dict()
    trackVelocitiesById = dict()
    for trk in trk3dArray.trks_list:
        trackPositionsById[trk.tracked_id] = numpy.array([trk.x, trk.y])
        trackVelocitiesById[trk.tracked_id] = numpy.array([trk.vx, trk.vy])

    # 對每一組，嘗試從歷史群組中找出相似的並延續群組 ID
    for clusterId, track_ids in sortedGroups:
        bestGroupIdAssignment = None
        trackIdSet = set(track_ids)
        for groupIdAssignment in trackGroups.groupIdAssignmentMemory:
            if groupIdAssignment.trackIds.issuperset(trackIdSet) or groupIdAssignment.trackIds.issubset(trackIdSet):
                currentCount = len(groupIdAssignment.trackIds)
                bestCount = None if bestGroupIdAssignment is None else len(bestGroupIdAssignment.trackIds)
                if bestGroupIdAssignment is None or currentCount > bestCount or (currentCount == bestCount and groupIdAssignment.createdAt < bestGroupIdAssignment.createdAt):
                    if groupIdAssignment.groupId not in assignedGroupIds:
                        bestGroupIdAssignment = groupIdAssignment

        groupId = None
        if bestGroupIdAssignment is not None:
            groupId = bestGroupIdAssignment.groupId

        if groupId is None or groupId in assignedGroupIds:
            groupId = trackGroups.largestGroupId + 1

        assignedGroupIds.append(groupId)

        groupExistsSince = trk3dArray.header.stamp.to_sec()
        groupIdAssignmentsToRemove = []
        for groupIdAssignment in trackGroups.groupIdAssignmentMemory:
            if set(track_ids) == groupIdAssignment.trackIds:
                groupExistsSince = min(groupIdAssignment.createdAt, groupExistsSince)
                groupIdAssignmentsToRemove.append(groupIdAssignment)
        for groupIdAssignment in groupIdAssignmentsToRemove:
            trackGroups.groupIdAssignmentMemory.remove(groupIdAssignment)

        trackGroups.groupIdAssignmentMemory.append(GroupIdAssignment(track_ids, groupId, groupExistsSince))
        if groupId > trackGroups.largestGroupId:
            trackGroups.largestGroupId = groupId

        # 若不發佈單人群組，就略過只有一人的情況
        if publishSinglePersonGroups or len(track_ids) > 1:
            # 計算群組重心與群組平均速度
            accumulatedPosition = numpy.array([0.0, 0.0])
            accumulatedVelocity = numpy.array([0.0, 0.0])
            activeTrackCount = 0
            for tid in track_ids:
                if tid in trackPositionsById:
                    accumulatedPosition += trackPositionsById[tid]
                    if tid in trackVelocitiesById:
                        accumulatedVelocity += trackVelocitiesById[tid]
                    activeTrackCount += 1

            # 計算群組平均位置
            centroid = accumulatedPosition / float(activeTrackCount)
            # 計算群組平均速度
            avgVelocity = accumulatedVelocity / float(activeTrackCount)

            trackedGroup = TrackedGroup()
            trackedGroup.age = rospy.Duration(max(0, trk3dArray.header.stamp.to_sec() - groupExistsSince))
            trackedGroup.group_id = remapGroupId(groupId)
            trackedGroup.track_ids = track_ids
            # 設定重心 (僅提供 x 與 y)
            trackedGroup.centerOfGravity.pose.position.x = centroid[0]
            trackedGroup.centerOfGravity.pose.position.y = centroid[1]
            trackedGroup.group_velocity.linear.x = avgVelocity[0]
            trackedGroup.group_velocity.linear.y = avgVelocity[1]

            # For testing purposes, set a constant velocity
            # trackedGroup.group_velocity.linear.x = -1.0
            # trackedGroup.group_velocity.linear.y = -1.0

            if len(track_ids) == 1:
                # 單人群組，用那個人的 radius
                uid = track_ids[0]
                person = next(trk for trk in trk3dArray.trks_list if trk.tracked_id == uid)
                trackedGroup.group_radius = person.radius
            else:
                # 多人群組，計算重心到最遠成員的距離
                max_dist = 0.0
                for uid in track_ids:
                    person = next(trk for trk in trk3dArray.trks_list if trk.tracked_id == uid)
                    dist = numpy.linalg.norm(numpy.array([person.x, person.y]) - centroid)
                    max_dist = max(max_dist, dist + person.radius)
                trackedGroup.group_radius = max(max_dist, 0.5)  # 確保半徑至少為0.5

            trackedGroups.append(trackedGroup)

    return trackedGroups


#########################################################################
# 發佈目前分群結果
#########################################################################
def publishGroups(groups, trk3dArray):
    msg = TrackedGroups()
    msg.header = trk3dArray.header
    msg.groups = groups
    pub.publish(msg)

def publishGroupMarkers(trackedGroups, trk3dArray):
    global last_group_ids
    marker_array = MarkerArray()
    current_group_ids = set()
    for i, group in enumerate(trackedGroups):
        # 圓形範圍 marker
        circle = Marker()
        circle.header.frame_id = trk3dArray.header.frame_id
        circle.header.stamp = trk3dArray.header.stamp
        circle.ns = "group"
        circle.id = group.group_id
        circle.type = Marker.CYLINDER
        circle.action = Marker.ADD
        circle.pose.position.x = group.centerOfGravity.pose.position.x
        circle.pose.position.y = group.centerOfGravity.pose.position.y
        circle.pose.position.z = 0
        circle.scale.x = group.group_radius * 2
        circle.scale.y = group.group_radius * 2
        circle.scale.z = 0.05
        circle.color.r = 0.1
        circle.color.g = 0.8
        circle.color.b = 0.2
        circle.color.a = 0.5
        marker_array.markers.append(circle)

        # 文字 label marker
        text = Marker()
        text.header.frame_id = trk3dArray.header.frame_id
        text.header.stamp = trk3dArray.header.stamp
        text.ns = "group_label"
        text.id = group.group_id
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.x = group.centerOfGravity.pose.position.x
        text.pose.position.y = group.centerOfGravity.pose.position.y
        text.pose.position.z = 0.3
        text.scale.z = 0.3
        text.color.r = 0.0
        text.color.g = 0.0
        text.color.b = 0.0
        text.color.a = 1.0
        text.text = f"Group {group.group_id}"
        marker_array.markers.append(text)

        current_group_ids.add(group.group_id)

    # 移除不再存在的 marker
    vanished_ids = last_group_ids - current_group_ids
    for gid in vanished_ids:
        delete_marker = Marker()
        delete_marker.header.frame_id = trk3dArray.header.frame_id
        delete_marker.header.stamp = trk3dArray.header.stamp
        delete_marker.ns = "group"
        delete_marker.id = gid
        delete_marker.action = Marker.DELETE
        marker_array.markers.append(delete_marker)

        delete_text = Marker()
        delete_text.header.frame_id = trk3dArray.header.frame_id
        delete_text.header.stamp = trk3dArray.header.stamp
        delete_text.ns = "group_label"
        delete_text.id = gid
        delete_text.action = Marker.DELETE
        marker_array.markers.append(delete_text)

    # 更新最後的群組 ID 集合
    last_group_ids = current_group_ids

    marker_pub.publish(marker_array)



#########################################################################
# 主函數：訂閱「trk3d_result」及 social relation 資料後進行分群
#########################################################################
def main():
    rospy.init_node("tracked_groups")

    # 初始化群組 ID 的記錄變數
    trackGroups.largestGroupId = -1
    trackGroups.groupIdAssignmentMemory = deque(maxlen=300)
    trackGroups.groupIdRemapping = deque(maxlen=50)
    trackGroups.largestPublishedGroupId = -1

    # 這裡 SocialRelations 的 topic 依然使用原有 topic，如有需要請一併修改
    trackedPersonsTopic = rospy.resolve_name("trk3d_result")
    socialRelationsTopic = rospy.resolve_name("social_relations")
    # socialRelationsTopic = rospy.resolve_name("long_term_social_relations")
    trackedGroupsTopic = rospy.resolve_name("tracked_groups")
    groupVisualizationTopic = rospy.resolve_name("tracked_groups_visualization")

    # 使用 message_filters 同步兩個 topic 資料
    trk3d_sub = message_filters.Subscriber(trackedPersonsTopic, Trk3DArray)
    socialRelations_sub = message_filters.Subscriber(socialRelationsTopic, SocialRelations)
    ts = message_filters.TimeSynchronizer([trk3d_sub, socialRelations_sub], 100)
    ts.registerCallback(newDataAvailable)

    rospy.loginfo("Subscribing to %s and %s" % (socialRelationsTopic, trackedPersonsTopic))

    global pub, marker_pub
    pub = rospy.Publisher(trackedGroupsTopic, TrackedGroups, queue_size=3)
    marker_pub = rospy.Publisher(groupVisualizationTopic, MarkerArray, queue_size=1)
    rospy.loginfo("Publishing tracked groups on %s" % (trackedGroupsTopic))

    rospy.spin()

if __name__ == '__main__':
    main()
