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
整合空間關係計算與長期關係更新的 ROS 節點
作者：你的名字（請自行修改）
"""

import os, sys, math, time, copy, numpy
import rospy, message_filters
from math import hypot, pi, fabs
from geometry_msgs.msg import Point, Vector3
from tf.transformations import euler_from_quaternion
# from spencer_tracking_msgs.msg import TrackedPersons, TrackedPerson
# from spencer_social_relation_msgs.msg import SocialRelations, SocialRelation
from ultralytics_ros.msg import Trk3DArray, Trk3D
from social_rules_selector.msg import SocialRelations, SocialRelation

# 載入 libsvm 的 Python 介面
# 請先確保已安裝 libsvm 並能正確 import svmutil
from libsvm.svmutil import svm_load_model, svm_predict

# ------------------------
# 全域參數與變數設定
# ------------------------
# 空間關係參數 (取自參數伺服器或預設值)
MAX_DISTANCE = None
MAX_SPEED_DIFFERENCE = None
MAX_ORIENTATION_DIFFERENCE = None
MIN_SPEED_TO_CONSIDER_ORIENTATION = None

# SVM model 相關變數
svm_model = None

# long term relation parameters
growthAmount    = None   # 關係成長速率（每秒）
slowDecayAmount = None   # 慢衰減速率（每秒）
fastDecayAmount = None   # 快衰減速率（每秒）
LT_MAX_DISTANCE = None   # 當距離超過此值則強制關係降為 0

# 用於長期平滑更新的關係儲存（key 為 (type, smaller_track_id, larger_track_id)）
stored_relations = dict()
last_data_received_at = None

# ROS Publisher (用於發佈長期關係訊息)
lt_pub = None
# （若需要也可發佈空間關係，可新增 publisher，此處示範僅發佈最終結果）
spatial_pub = None


# ------------------------
# 工具函數 (角度處理)
# ------------------------
def set_angle_to_range(alpha, min_val):
    """將角度 alpha 轉換至 [min_val, min_val+2*pi) 區間"""
    while alpha >= min_val + 2.0 * pi:
        alpha -= 2.0 * pi
    while alpha < min_val:
        alpha += 2.0 * pi
    return alpha

def diff_angle_unwrap(alpha1, alpha2):
    """
    計算兩角度（弧度）之間的最小差值（考慮循環性）
    """
    alpha1 = set_angle_to_range(alpha1, 0)
    alpha2 = set_angle_to_range(alpha2, 0)
    delta = alpha1 - alpha2
    if alpha1 > alpha2:
        while delta > pi:
            delta -= 2.0 * pi
    elif alpha2 > alpha1:
        while delta < -pi:
            delta += 2.0 * pi
    return delta

# ------------------------
# 空間關係計算函數
# ------------------------
def compute_spatial_relations(trk3d_msg):
    """
    根據 trk3d_msg (Trk3DArray) 中的所有行人(trks_list)，
    對每對行人計算距離、速度差與方向差，
    並利用 SVM 預測兩人間的空間關係機率 (TYPE_SPATIAL)。
    """
    spatial_relations = SocialRelations()
    spatial_relations.header = trk3d_msg.header

    trks = trk3d_msg.trks_list
    num_tracks = len(trks)
    for i in range(num_tracks):
        # print("Processing track %d/%d" % (i+1, num_tracks))
        for j in range(i+1, num_tracks):

            t1 = trks[i]
            t2 = trks[j]
            # 位置
            x1, y1 = t1.x, t1.y
            x2, y2 = t2.x, t2.y

            # 速度 (只取水平分量)
            vx1, vy1 = t1.vx, t1.vy
            vx2, vy2 = t2.vx, t2.vy
            speed1 = hypot(vx1, vy1)
            speed2 = hypot(vx2, vy2)

            # 方向 (若速度不夠大，則不採用 yaw)
            yaw1 = 0.0
            yaw2 = 0.0
            if speed1 >= MIN_SPEED_TO_CONSIDER_ORIENTATION:
                yaw1 = t1.yaw  # 假設 yaw 已為弧度
            if speed2 >= MIN_SPEED_TO_CONSIDER_ORIENTATION:
                yaw2 = t2.yaw

            # 計算特徵
            distance = hypot(x1 - x2, y1 - y2)
            deltaspeed = fabs(speed1 - speed2)
            deltaangle = fabs(diff_angle_unwrap(yaw1, yaw2))

            # 若任一特徵超出設定閥值，直接給定較低的正向機率
            if distance > MAX_DISTANCE or deltaspeed > MAX_SPEED_DIFFERENCE or deltaangle > MAX_ORIENTATION_DIFFERENCE:
                positiveProbability = 0.1
            else:
                # 構造 SVM 特徵向量 (注意：index 從1開始)
                feature_vector = {1: distance, 2: deltaspeed, 3: deltaangle}
                # svm_predict_probability 傳入 list 格式：此處只預測一筆資料
                p_label, p_acc, p_vals = svm_predict([0], [feature_vector], svm_model, options='-b 1 -q')
                # 假設 p_vals[0][0] 為正向關係機率
                positiveProbability = p_vals[0][0]

            # 建立 SocialRelation 資訊
            relation = SocialRelation()
            relation.type = SocialRelation.TYPE_SPATIAL  # 請確認此常數是否正確
            relation.strength = positiveProbability
            # 以 tracked_id 做為識別 (依照 int32 定義)
            relation.track1_id = t1.tracked_id
            relation.track2_id = t2.tracked_id

            spatial_relations.elements.append(relation)

    return spatial_relations

# ------------------------
# 長期關係平滑更新函數
# ------------------------
def update_long_term_relations(trk3d_msg, inputRelations, current_time):
    """
    根據最新的空間關係(inputRelations)與行人資料(trk3d_msg)
    平滑更新長期關係。
    """
    global stored_relations, last_data_received_at

    # 取得時間差 (秒)
    deltaTime = (current_time - last_data_received_at).to_sec() if last_data_received_at is not None else 0.0
    last_data_received_at = current_time

    # 移除不再存在的 track，利用 trks_list 中的 tracked_id
    current_ids = set([trk.tracked_id for trk in trk3d_msg.trks_list])
    obsolete_keys = []
    for key in stored_relations.keys():
        if key[1] not in current_ids or key[2] not in current_ids:
            obsolete_keys.append(key)
    for key in obsolete_keys:
        del stored_relations[key]

    # 建立行人位置字典 {tracked_id: numpy.array([x, y])}
    tracked_positions = {}
    for trk in trk3d_msg.trks_list:
        tracked_positions[trk.tracked_id] = numpy.array([trk.x, trk.y])

    # 準備更新後的長期關係輸出訊息
    outputRelations = SocialRelations()
    outputRelations.header = inputRelations.header

    # 更新每一組空間關係
    for input_relation in inputRelations.elements:
        id1, id2 = input_relation.track1_id, input_relation.track2_id
        smaller_id = min(id1, id2)
        larger_id  = max(id1, id2)
        key = (input_relation.type, smaller_id, larger_id)
        stored_strength = stored_relations.get(key, 0.0)

        target_strength = input_relation.strength
        decay = slowDecayAmount

        # 若兩行人距離超過設定 LT_MAX_DISTANCE，則強制 target_strength 為 0 並使用快衰減
        if LT_MAX_DISTANCE is not None:
            pos1 = tracked_positions.get(smaller_id)
            pos2 = tracked_positions.get(larger_id)
            if pos1 is not None and pos2 is not None:
                distance = numpy.linalg.norm(pos1 - pos2)
                if distance > LT_MAX_DISTANCE:
                    target_strength = 0.0
                    decay = fastDecayAmount

        # 平滑更新：上升使用成長、下降使用衰減
        if target_strength >= stored_strength:
            stored_strength += growthAmount * deltaTime
            if stored_strength > target_strength:
                stored_strength = target_strength
        else:
            stored_strength -= decay * deltaTime
            if stored_strength < target_strength:
                stored_strength = target_strength

        stored_relations[key] = stored_strength

        # 複製輸入關係，更新 strength 值後加入輸出
        new_relation = copy.deepcopy(input_relation)
        new_relation.strength = stored_strength
        outputRelations.elements.append(new_relation)

    return outputRelations

# ------------------------
# 回呼主流程：結合空間關係計算與長期關係更新
# ------------------------
def data_callback(trk3d_msg, dummy_msg):
    """
    同步接收 trk3d_msg（包含 header）後：
      1. 計算空間關係
      2. 更新長期關係平滑結果
      3. 發佈更新後的長期關係訊息
    """
    global last_data_received_at

    current_time = trk3d_msg.header.stamp
    if last_data_received_at is None:
        last_data_received_at = current_time

    # 計算空間關係（相當於 spatial_relations.cpp 的運算）
    spatial_relations = compute_spatial_relations(trk3d_msg)
    # 此處若需要也可發佈空間關係，另外發佈者可另外設定
    spatial_pub.publish(spatial_relations)

    # 更新長期關係 (long_term_relations)
    lt_relations = update_long_term_relations(trk3d_msg, spatial_relations, current_time)
    
    # 發佈更新後的長期關係
    lt_pub.publish(lt_relations)

# ------------------------
# 主函數
# ------------------------
def main():
    global MAX_DISTANCE, MAX_SPEED_DIFFERENCE, MAX_ORIENTATION_DIFFERENCE, MIN_SPEED_TO_CONSIDER_ORIENTATION
    global growthAmount, slowDecayAmount, fastDecayAmount, LT_MAX_DISTANCE, svm_model, lt_pub, spatial_pub

    rospy.init_node('social_relations_node')

    # 讀取參數（預設值參考原始程式）
    MAX_DISTANCE = rospy.get_param("~max_distance", 3.0)
    MAX_SPEED_DIFFERENCE = rospy.get_param("~max_speed_difference", 1.0)
    MAX_ORIENTATION_DIFFERENCE = rospy.get_param("~max_orientation_difference", pi/4)
    MIN_SPEED_TO_CONSIDER_ORIENTATION = rospy.get_param("~min_speed_to_consider_orientation", 0.1)

    growthAmount    = rospy.get_param("~growth", 0.5/3.5)
    slowDecayAmount = rospy.get_param("~slow_decay", 0.5/40.0)
    fastDecayAmount = rospy.get_param("~fast_decay", 0.5/5.0)
    LT_MAX_DISTANCE = rospy.get_param("~lt_max_distance", 4.0)

    # 載入 SVM 模型 (請設定正確的 model 檔案路徑)
    model_path = rospy.get_param("~model_path", "../models")
    svm_filename = rospy.get_param("~svm_filename", os.path.join(model_path, "groups_probabilistic_small.model"))
    if not os.path.isfile(svm_filename):
        rospy.logfatal("SVM model file %s not found!", svm_filename)
        sys.exit(1)

    # 載入 SVM 模型 (此模型需支援機率預測)
    svm_model = svm_load_model(svm_filename)
    if svm_model is None:
        rospy.logfatal("Failed to load SVM model from %s", svm_filename)
        sys.exit(1)
    rospy.loginfo("Loaded SVM model from %s", svm_filename)

    # 建立 Publisher，發佈長期關係結果 (topic 可依需求調整)
    spatial_pub = rospy.Publisher(rospy.resolve_name("social_relations"), SocialRelations, queue_size=3)
    lt_pub = rospy.Publisher(rospy.resolve_name("long_term_social_relations"), SocialRelations, queue_size=3)

    # 由於空間關係完全能從 trackedPersons 資料計算，
    # 這裡用 message_filters.Subscriber 同時訂閱 trackedPersons 與自己 dummy 一份 (以取得 header stamp)
    tracked_sub = message_filters.Subscriber(rospy.resolve_name("trk3d_result"), Trk3DArray)
    # 為簡化，dummy_relations_sub 也訂閱 trackedPersons (僅用 header)，可依需求替換
    dummy_relations_sub = message_filters.Subscriber(rospy.resolve_name("trk3d_result"), Trk3DArray)
    
    ts = message_filters.TimeSynchronizer([tracked_sub, dummy_relations_sub], 5)
    ts.registerCallback(data_callback)
    
    rospy.loginfo("Social Relations Node started. Listening to tracked persons data ...")
    rospy.spin()

if __name__ == '__main__':
    main()
