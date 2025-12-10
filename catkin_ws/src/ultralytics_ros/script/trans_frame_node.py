#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import rospy
import tf2_ros
import tf2_geometry_msgs
import tf.transformations as tf_trans
from std_msgs.msg import Header
from geometry_msgs.msg import Point, PointStamped, Quaternion
from visualization_msgs.msg import Marker, MarkerArray
from ultralytics_ros.msg import Det3DArray, Trk3DArray, Trk3D

class SimpleKalman:
    """針對每個行人ID的簡易2D卡爾曼濾波 (x, y, vx, vy)"""
    def __init__(self, x, y, radius=0.3):
        self.x = np.array([[x], [y], [0.0], [0.0]])  # 狀態向量 [x, y, vx, vy]
        self.P = np.eye(4) * 1.0                     # 狀態不確定度
        self.Q = np.eye(4) * 0.05                    # 過程雜訊
        self.R = np.eye(2) * 0.2                     # 觀測雜訊
        self.last_update = rospy.Time.now()
        self.last_predict = self.last_update
        self.has_measurement = True
        self.radius = radius  # 初始半徑
        self.age = 0
        self.validity = False

    def predict(self, dt):
        # 狀態轉移矩陣
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1,  0],
                      [0, 0, 0,  1]])
        self.x = F @ self.x # 預測下一狀態
        self.P = F @ self.P @ F.T + self.Q # 更新不確定度
        self.last_predict = rospy.Time.now()

    def update(self, z): 
        # Ref : https://chih-sheng-huang821.medium.com/%E7%B0%A1%E6%98%93%E4%BB%8B%E7%B4%B9%E5%8D%A1%E7%88%BE%E6%9B%BC%E6%BF%BE%E6%B3%A2-kalman-filter-1b041e371fe6
        """z = [x, y]"""
        # 觀測矩陣
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])
        y = z - H @ self.x # 預測值和量測值的差異
        # 計算卡爾曼增益
        S = H @ self.P @ H.T + self.R

        # --- Mahalanobis distance gate ---
        d = float(y.T @ np.linalg.inv(S) @ y)
        gating_threshold = 10.0  # 可調，越大越寬鬆

        if d > gating_threshold and self.validity:
            # 量測太離譜 → 不更新，視同沒有測到
            self.has_measurement = False
            return self.get_state()

        # --- Normal Kalman update ---
        K = self.P @ H.T @ np.linalg.inv(S)
        # 更新狀態
        self.x = self.x + K @ y
        # 更新不確定度
        I = np.eye(4)
        self.P = (I - K @ H) @ self.P

        self.last_update = rospy.Time.now()
        self.has_measurement = True
        self.age += 1
        if self.age > 5:
            self.validity = True

    def step(self, dt, measurement=None):
        """measurement: (x, y) or None"""
        if measurement is not None:
            self.predict(dt)
            self.update(np.array([[measurement[0]], [measurement[1]]]))
        else:
            vx, vy = self.x[2, 0], self.x[3, 0]
            speed = math.hypot(vx, vy)
            if speed > 0.25:
                self.predict(dt)
            self.has_measurement = False
        return self.get_state()

    def get_state(self):
        return self.x.flatten()  # [x, y, vx, vy]


class Yolo3DTransformNode:
    def __init__(self):
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(2.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.pub_pose = rospy.Publisher("transformed_bbox_center", Point, queue_size=10)
        self.pub_trk = rospy.Publisher("trk3d_result", Trk3DArray, queue_size=10)
        self.pub_marker = rospy.Publisher("trk3d_visual", MarkerArray, queue_size=10)

        rospy.Subscriber("yolo_3d_result", Det3DArray, self.detection_callback)

        self.trackers = {}  # {id: SimpleKalman}
        self.last_time = None
        self.duration = 0.1

        # 可調參數
        self.predict_only_timeout = 2.0  # 沒偵測時允許預測多久
        self.remove_timeout = 2.0        # 超過多久沒更新就刪除

    def detection_callback(self, msg):
        trk3d_array = Trk3DArray()
        trk3d_array.header = Header()
        trk3d_array.header.frame_id = "odom"
        trk3d_array.header.stamp = msg.header.stamp
        marker_array = MarkerArray()

        current_time = rospy.Time.now()
        if self.last_time is None:
            self.last_time = current_time
            return
        self.duration = (current_time - self.last_time).to_sec()
        if self.duration <= 0.04:
            self.duration = 0.04
        self.last_time = current_time

        updated_ids = set()

        # --- 更新有偵測到的ID ---
        for det in msg.detections:
            if det.tracked_id == -1:
                continue  # 跳過無效ID

            bbox_center = det.center.position
            bbox_center_stamped = PointStamped()
            bbox_center_stamped.header = msg.header
            bbox_center_stamped.point = Point(bbox_center.x, bbox_center.y, bbox_center.z)

            try:
                bbox_center_odom = self.tf_buffer.transform(
                    bbox_center_stamped, "odom", rospy.Duration(1.0)
                ).point

                obj_id = det.tracked_id
                if obj_id not in self.trackers:
                    init_r = max(det.size.x, det.size.y) / 2
                    self.trackers[obj_id] = SimpleKalman(bbox_center_odom.x, bbox_center_odom.y, init_r)

                kf = self.trackers[obj_id]
                x, y, vx, vy = kf.step(self.duration, (bbox_center_odom.x, bbox_center_odom.y))

                r_new = max(det.size.x, det.size.y) / 2
                r_min = 0.2       # 最小半徑限制
                alpha_grow = 0.3  # 放大時的更新比例
                alpha_shrink = 0.1  # 縮小時的更新比例
                
                if r_new > kf.radius:
                    kf.radius = kf.radius * (1 - alpha_grow) + r_new * alpha_grow
                else:
                    kf.radius = kf.radius * (1 - alpha_shrink) + r_new * alpha_shrink

                kf.radius = max(kf.radius, r_min)

                updated_ids.add(obj_id)

                if kf.validity:
                    self._publish_tracked_object(marker_array, trk3d_array, det, x, y, vx, vy)
                self.pub_pose.publish(bbox_center_odom)

            except (tf2_ros.LookupException, tf2_ros.ExtrapolationException) as ex:
                rospy.logwarn_throttle(2.0, "TF transform failed: %s", ex)

        # --- 沒偵測到的ID，繼續預測 ---
        for obj_id, kf in list(self.trackers.items()):
            if obj_id not in updated_ids:
                time_since_update = (current_time - kf.last_update).to_sec()
                if time_since_update < self.predict_only_timeout:
                    x, y, vx, vy = kf.step(self.duration, measurement=None)
                    if kf.validity:
                        self._publish_predicted_object(marker_array, trk3d_array, obj_id, x, y, vx, vy)
                elif time_since_update > self.remove_timeout:
                    del self.trackers[obj_id]

        # --- 發布結果 ---
        self.pub_trk.publish(trk3d_array)
        self.pub_marker.publish(marker_array)

    def _publish_tracked_object(self, marker_array, trk3d_array, det, x, y, vx, vy):
        trk3d = Trk3D()
        trk3d.tracked_id = det.tracked_id
        trk3d.class_name = det.class_name
        trk3d.x = x
        trk3d.y = y
        norm = math.hypot(vx, vy)
        if norm < 0.25:
            vx, vy = 0.0, 0.0
        trk3d.vx = vx
        trk3d.vy = vy
        trk3d.yaw = math.atan2(vy, vx)
        trk3d.radius = self.trackers[det.tracked_id].radius
        trk3d.confidence = det.score
        trk3d_array.trks_list.append(trk3d)

        marker = self._make_marker(det.tracked_id, x, y, trk3d.yaw, trk3d.radius, [0.1, 0.9, 0.1])
        marker_array.markers.append(marker)

    def _publish_predicted_object(self, marker_array, trk3d_array, obj_id, x, y, vx, vy):
        """沒有偵測更新時的預測結果（顏色不同）"""
        trk3d = Trk3D()
        trk3d.tracked_id = obj_id
        trk3d.class_name = "predicted"
        trk3d.x = x
        trk3d.y = y
        norm = math.hypot(vx, vy)
        if norm < 0.25:
            vx, vy = 0.0, 0.0
        trk3d.vx = vx
        trk3d.vy = vy
        trk3d.yaw = math.atan2(vy, vx)
        trk3d.radius = self.trackers[obj_id].radius
        trk3d.confidence = 0.0
        trk3d_array.trks_list.append(trk3d)

        marker = self._make_marker(obj_id, x, y, trk3d.yaw, trk3d.radius, [0.9, 0.9, 0.1])
        marker_array.markers.append(marker)

    def _make_marker(self, obj_id, x, y, yaw, radius, rgb):
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "people"
        marker.id = obj_id
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.8
        quat = tf_trans.quaternion_from_euler(0, 0, yaw)
        marker.pose.orientation = Quaternion(*quat)
        marker.scale.x = radius * 2.0
        marker.scale.y = radius * 2.0
        marker.scale.z = 1.6
        marker.color.a = 0.6
        marker.color.r = rgb[0]
        marker.color.g = rgb[1]
        marker.color.b = rgb[2]
        marker.lifetime = rospy.Duration(1.0)
        return marker


if __name__ == '__main__':
    rospy.init_node('yolo_3d_transform_node')
    node = Yolo3DTransformNode()
    rospy.spin()
