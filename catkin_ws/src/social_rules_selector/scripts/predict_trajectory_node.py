#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from social_rules_selector.msg import PredictTrajs, PredictTraj
from social_rules_selector.msg import TrackedGroups
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
import os
import sys
import colorsys  # 用於 HSV to RGB 轉換

sys.path.append(os.path.join(os.path.dirname(__file__)))
from CVM import predict_trajectory

class PredictTrajectoryNode(object):
    def __init__(self):
        rospy.init_node("predict_trajectory_node", anonymous=True)
        self.input_topic = rospy.get_param("~input_topic", "/tracked_groups")
        self.output_topic = rospy.get_param("~output_topic", "/predicted_trajectory")
        self.predict_time = rospy.get_param("~predict_time", 1.0)
        self.dt = rospy.get_param("~predict_dt", 0.1)

        self.pub = rospy.Publisher(self.output_topic, PredictTrajs, queue_size=1)
        self.vis_pub = rospy.Publisher("/visualization_predicted_trajectories", MarkerArray, queue_size=1)
        self.sub = rospy.Subscriber(self.input_topic, TrackedGroups, self.groups_callback)

        rospy.loginfo("PredictTrajectoryNode 初始化完成，訂閱：%s，發布：%s", self.input_topic, self.output_topic)

    def groups_callback(self, msg):
        pred_msg = predict_trajectory(msg, self.predict_time, self.dt)
        if pred_msg is not None:
            self.pub.publish(pred_msg)
            self.publish_visualization(pred_msg)
        else:
            rospy.logwarn("PredictTrajectoryNode Failed : No prediction result")

    def publish_visualization(self, pred_msg):
        marker_array = MarkerArray()
        for i, traj in enumerate(pred_msg.predicted_trajs):
            marker = Marker()
            marker.header = pred_msg.header
            marker.ns = "predicted_trajectory"
            marker.id = traj.group_id  # 每個 group 不同 marker
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.05  # 線寬
            marker.pose.orientation.w = 1.0

            # 顏色：根據 group_id hash 一個 HSV，轉 RGB
            r, g, b = self.color_from_id(traj.group_id)
            marker.color.r = r
            marker.color.g = g
            marker.color.b = b
            marker.color.a = 1.0

            # 預測軌跡點
            marker.points = []
            for pose in traj.predicted_trajectory:
                pt = Point()
                pt.x = pose.pose.position.x
                pt.y = pose.pose.position.y
                pt.z = 0.1
                marker.points.append(pt)

            # lifetime：軌跡會自動消失
            marker.lifetime = rospy.Duration(self.dt * len(traj.predicted_trajectory))

            marker_array.markers.append(marker)

        self.vis_pub.publish(marker_array)

    def color_from_id(self, id):
        """
        將 group_id 映射到獨特的 RGB 顏色。
        使用 HSV 色彩空間做分佈，避免顏色重複。
        """
        hue = (hash(id) % 360) / 360.0  # hue in [0,1)
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        return r, g, b

if __name__ == '__main__':
    try:
        node = PredictTrajectoryNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
