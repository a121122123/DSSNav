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
    def __init__(self, x, y, radius=0.2):
        self.x = np.array([[x], [y], [0.0], [0.0]])  # 狀態向量 [x, y, vx, vy]
        self.P = np.eye(4) * 1.0                     # 狀態不確定度
        self.P[3:, 3:] *= 1000                       # 增加速度的不確定度
        self.Q = np.eye(4) * 0.05                    # 過程雜訊
        self.R_default = np.eye(2) * 0.2             # 預設觀測雜訊
        self.last_update = rospy.Time.now()
        self.last_predict = self.last_update
        self.has_measurement = True
        self.radius = radius
        self.age = 0
        self.validity = False
        self.vel_conf_counter = 0
        self.vel_conf = False

    def predict(self, dt):
        # 狀態轉移矩陣 (假設常速模型)
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1,  0],
                      [0, 0, 0,  1]])
        self.x = F @ self.x # 預測下一狀態
        self.P = F @ self.P @ F.T + self.Q # 更新不確定度
        self.last_predict = rospy.Time.now()

    def update(self, z, dt, near_edge=False):
        # Ref : https://chih-sheng-huang821.medium.com/%E7%B0%A1%E6%98%93%E4%BB%8B%E7%B4%B9%E5%8D%A1%E7%88%BE%E6%9B%BC%E6%BF%BE%E6%B3%A2-kalman-filter-1b041e371fe6
        """
        z = [x, y]
        near_edge: 如果在邊緣，我們會大幅增加觀測雜訊 R，讓濾波器更相信預測值
        """
        # 觀測矩陣 (只觀測位置，不直接觀測速度)
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])
        
        # 邊緣邏輯：若在邊緣，將觀測雜訊放大 20 倍，防止中心點偏移拉走狀態
        R = self.R_default * (100.0 if near_edge else 1.0)
        
        y = z - H @ self.x # 觀測誤差, 觀測值與預測值的差距

        # 如果觀測值與預測值的速度差距過大，則認為這次觀測不可靠，增加觀測雜訊
        is_jump = False
        if dt > 0 and (np.linalg.norm(y) / dt) > 3.0: 
            # R = self.R_default * 500.0
            is_jump = True

        # 計算創新協方差 S
        S = H @ self.P @ H.T + R

        # Mahalanobis distance gate
        try:
            inv_S = np.linalg.inv(S)
            d = float(y.T @ inv_S @ y)
        except np.linalg.LinAlgError:
            return self.get_state()

        gating_threshold = 2.5 # 可調，根據實驗調整，過大會接受錯誤觀測，過小會過度拒絕正確觀測
        # rospy.loginfo("Mahalanobis distance: %.3f (threshold: %.1f)", d, gating_threshold)
        if d > gating_threshold and self.validity:
            # 如果觀測值與預測值差距過大，且目前狀態已經穩定(validity)，則拒絕這次更新，保持原狀態不變
            self.has_measurement = False
            return self.get_state()

        # 計算卡爾曼增益 K
        K = self.P @ H.T @ inv_S
        
        # 如果在邊緣，凍結速度更新，只微調位置
        if near_edge or is_jump:
            K[2:, :] = 0.0

        # 更新狀態和不確定度
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P

        self.last_update = rospy.Time.now()
        self.has_measurement = True
        
        if self.age < 5:
            self.age += 1
        else:
            self.validity = True
        
        speed = math.hypot(self.x[2, 0], self.x[3, 0])
        if speed > 0.25:
            self.vel_conf_counter += 1
        else:
            self.vel_conf_counter = max(0, self.vel_conf_counter - 1)
        self.vel_conf = self.vel_conf_counter >= 5

    def step(self, dt, measurement=None, bbox2d_center_x=None, bbox2d_size_x=None):
        """執行一次完整的 Kalman 步驟，包含預測和更新，並根據是否在邊緣調整更新權重
        measurement: (x, y) 或 None
        bbox2d_center_x: 用於判斷是否在邊緣的2D框中心x座標
        """
        near_edge = self.is_near_fov_edge_bbox_center(bbox2d_center_x, image_width=640, threshold=0.2)
        # TODO: 可以改成 is_near_fov_edge_bbox，考慮框的大小
        # near_edge = self.is_near_fov_edge_bbox(bbox2d_center_x, bbox2d_size_x, image_width=640, margin=5)

        if measurement is not None:
            self.predict(dt)
            # 在邊緣時，update 會自動根據 near_edge 降權
            self.update(np.array([[measurement[0]], [measurement[1]]]), dt, near_edge=near_edge)
        else:
            # 沒偵測到時，若先前有速度則維持慣性
            vx, vy = self.x[2, 0], self.x[3, 0]
            speed = math.hypot(vx, vy)
            if speed > 0.25: 
                self.predict(dt)
            self.has_measurement = False
        
        return self.get_state(), near_edge

    def get_state(self):
        return self.x.flatten() # [x, y, vx, vy]
    
    def is_near_fov_edge_bbox_center(self, bbox2d_center_x, image_width, threshold=0.15):
        """判斷2D框中心是否接近圖像邊緣，threshold為邊緣區域占圖像寬度的比例"""
        if bbox2d_center_x is None:
            return False
        edge_limit = image_width * threshold
        return bbox2d_center_x < edge_limit or bbox2d_center_x > (image_width - edge_limit)
    
    def is_near_fov_edge_bbox(self, bbox2d_center_x, bbox2d_size_x, image_width, margin=5):
        """判斷2D框是否接近圖像邊緣，margin為距離邊緣的像素閾值"""
        if bbox2d_center_x is None or bbox2d_size_x is None:
            return False
        if bbox2d_center_x - bbox2d_size_x/2 < margin or bbox2d_center_x + bbox2d_size_x/2 > (image_width - margin):
            return True
        return False


class Yolo3DTransformNode:
    def __init__(self):
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(2.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.pub_trk = rospy.Publisher("trk3d_result", Trk3DArray, queue_size=10)
        self.pub_marker = rospy.Publisher("trk3d_visual", MarkerArray, queue_size=10)

        rospy.Subscriber("yolo_3d_result", Det3DArray, self.detection_callback)

        self.trackers = {}
        self.last_time = None
        self.predict_only_timeout = 2.0
        self.remove_timeout = 2.5

    def detection_callback(self, msg):
        current_time = rospy.Time.now()
        if self.last_time is None:
            self.last_time = current_time
            return
        
        duration = (current_time - self.last_time).to_sec()
        self.last_time = current_time

        trk3d_array = Trk3DArray()
        trk3d_array.header = Header(frame_id="odom", stamp=msg.header.stamp)
        marker_array = MarkerArray()
        updated_ids = set()

        for det in msg.detections:
            if det.tracked_id == -1: continue # 只處理有追蹤ID的目標

            try:
                # 座標轉換至 odom
                ps = PointStamped(header=msg.header, point=det.center.position)
                bbox_odom = self.tf_buffer.transform(ps, "odom", rospy.Duration(0.1)).point

                obj_id = det.tracked_id
                if obj_id not in self.trackers:
                    init_r = max(det.size.x, det.size.y) / 2
                    self.trackers[obj_id] = SimpleKalman(bbox_odom.x, bbox_odom.y, init_r)

                kf = self.trackers[obj_id]
                
                # 執行 Kalman Step (包含邊緣判斷邏輯)
                state, near_edge = kf.step(duration, (bbox_odom.x, bbox_odom.y), det.bbox2d_center.x, det.bbox2d_size_x)
                x, y, vx, vy = state

                # --- 半徑更新優化：邊緣時凍結半徑 ---
                if not near_edge:
                    r_new = max(det.size.x, det.size.y) / 2
                    alpha = 0.3 if r_new > kf.radius else 0.1
                    kf.radius = kf.radius * (1 - alpha) + r_new * alpha
                    kf.radius = np.clip(kf.radius, 0.2, 0.45)

                updated_ids.add(obj_id)

                if kf.validity:
                    self._add_to_msg(marker_array, trk3d_array, obj_id, det.class_name, x, y, vx, vy, kf.radius, [0.1, 0.9, 0.1], det.score)

            except Exception as ex:
                rospy.logwarn_throttle(2.0, f"TF Transform error: {ex}")

        # --- 處理遺失目標 (預測模式) ---
        for obj_id, kf in list(self.trackers.items()):
            if obj_id not in updated_ids:
                time_since_update = (current_time - kf.last_update).to_sec()
                if time_since_update < self.predict_only_timeout:
                    state, _ = kf.step(duration, measurement=None)
                    x, y, vx, vy = state
                    if kf.validity:
                        self._add_to_msg(marker_array, trk3d_array, obj_id, "predicted", x, y, vx, vy, kf.radius, [0.9, 0.9, 0.1], 0.0)
                elif time_since_update > self.remove_timeout:
                    del self.trackers[obj_id]

        # 發布追蹤結果和可視化
        self.pub_trk.publish(trk3d_array)
        self.pub_marker.publish(marker_array)

    def _add_to_msg(self, marker_array, trk3d_array, obj_id, cls, x, y, vx, vy, radius, rgb, score):
        trk = Trk3D()
        trk.tracked_id = obj_id
        trk.class_name = cls
        trk.x, trk.y = x, y
        # 低速過濾
        norm = math.hypot(vx, vy)
        trk.vx, trk.vy = (vx, vy) if norm > 0.25 else (0.0, 0.0)
        trk.yaw = math.atan2(vy, vx) if norm > 0.25 else 0.0
        trk.radius = radius
        trk.confidence = score
        trk3d_array.trks_list.append(trk)

        # 可視化
        m = Marker()
        m.header.frame_id = "odom"
        m.header.stamp = rospy.Time.now()
        m.ns = "people"; m.id = obj_id
        m.type = Marker.CYLINDER; m.action = Marker.ADD
        m.pose.position = Point(x, y, 0.8)
        q = tf_trans.quaternion_from_euler(0, 0, trk.yaw)
        m.pose.orientation = Quaternion(*q)
        m.scale.x = m.scale.y = radius * 2.0
        m.scale.z = 1.6
        m.color.a = 0.6; m.color.r, m.color.g, m.color.b = rgb
        m.lifetime = rospy.Duration(1.0)
        marker_array.markers.append(m)

if __name__ == '__main__':
    rospy.init_node('yolo_3d_transform_node')
    Yolo3DTransformNode()
    rospy.spin()









# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import math
# import numpy as np
# import rospy
# import tf2_ros
# import tf2_geometry_msgs
# import tf.transformations as tf_trans
# from std_msgs.msg import Header
# from geometry_msgs.msg import Point, PointStamped, Quaternion
# from visualization_msgs.msg import Marker, MarkerArray
# from ultralytics_ros.msg import Det3DArray, Trk3DArray, Trk3D

# class SimpleKalman:
#     """針對每個行人ID的簡易2D卡爾曼濾波 (x, y, vx, vy)"""
#     def __init__(self, x, y, radius=0.3):
#         self.x = np.array([[x], [y], [0.0], [0.0]])  # 狀態向量 [x, y, vx, vy]
#         self.P = np.eye(4) * 1.0                     # 狀態不確定度
#         self.P[3:, 3:] *= 1000                       # 增加速度的不確定度
#         self.Q = np.eye(4) * 0.05                    # 過程雜訊
#         self.R_default = np.eye(2) * 0.2                     # 觀測雜訊
#         self.last_update = rospy.Time.now()
#         self.last_predict = self.last_update
#         self.has_measurement = True
#         self.radius = radius  # 初始半徑
#         self.age = 0
#         self.validity = False
#         self.vel_conf_counter = 0
#         self.vel_conf = False

#     def predict(self, dt):
#         # 狀態轉移矩陣
#         F = np.array([[1, 0, dt, 0],
#                       [0, 1, 0, dt],
#                       [0, 0, 1,  0],
#                       [0, 0, 0,  1]])
#         self.x = F @ self.x # 預測下一狀態
#         self.P = F @ self.P @ F.T + self.Q # 更新不確定度
#         self.last_predict = rospy.Time.now()

#     def update(self, z, freeze_velocity=False): 
#         # Ref : https://chih-sheng-huang821.medium.com/%E7%B0%A1%E6%98%93%E4%BB%8B%E7%B4%B9%E5%8D%A1%E7%88%BE%E6%9B%BC%E6%BF%BE%E6%B3%A2-kalman-filter-1b041e371fe6
#         """z = [x, y]"""
#         # 觀測矩陣
#         H = np.array([[1, 0, 0, 0],
#                       [0, 1, 0, 0]])
#         y = z - H @ self.x # 預測值和量測值的差異
#         if freeze_velocity:
#             # rospy.logwarn("Measurement near FOV edge, freezing velocity update and increasing measurement noise.")
#             R = self.R_default * 50.0  # 增加量測不確定度，讓卡爾曼增益更小
#         else:
#             R = self.R_default

#         # 計算卡爾曼增益
#         S = H @ self.P @ H.T + R

#         # --- Mahalanobis distance gate --- 
#         d = float(y.T @ np.linalg.inv(S) @ y)
#         gating_threshold = 10.0  # 可調，越大越寬鬆

#         if d > gating_threshold and self.validity:
#             # 量測太離譜 → 不更新，視同沒有測到
#             self.has_measurement = False
#             if self.age > 0:
#                 self.age -= 1
#             if self.age < 5:
#                 self.validity = False
#             return self.get_state()
#         # ---------------------------------

#         # --- Normal Kalman update ---
#         K = self.P @ H.T @ np.linalg.inv(S)
#         # 凍結速度不更新
#         if freeze_velocity:
#             K[2:, :] = 0.0
#         # 更新狀態
#         self.x = self.x + K @ y
#         # 更新不確定度
#         I = np.eye(4)
#         self.P = (I - K @ H) @ self.P

#         self.last_update = rospy.Time.now()
#         self.has_measurement = True
#         if self.age < 5:
#             self.age += 1
#             self.validity = False
#         else:
#             self.validity = True
        
#         speed = math.hypot(self.x[2, 0], self.x[3, 0])
#         if speed > 0.25:
#             self.vel_conf_counter += 1
#         else:
#             self.vel_conf_counter = max(0, self.vel_conf_counter - 1)
#         self.vel_conf = self.vel_conf_counter >= 5

#     def step(self, dt, measurement=None, bbox2d_center_x=None):
#         """measurement: (x, y) or None"""
#         near_edge = self.is_near_fov_edge(bbox2d_center_x, image_width=640, threshold=0.15)
#         if measurement is not None:
#             # freeze_velocity = near_edge # and not self.vel_conf
#             if near_edge:
#                 self.predict(dt)
#                 self.has_measurement = True
#             else:
#                 self.predict(dt)
#                 self.update(np.array([[measurement[0]], [measurement[1]]]), freeze_velocity=near_edge)
#         else:
#             vx, vy = self.x[2, 0], self.x[3, 0]
#             speed = math.hypot(vx, vy)
#             if speed > 0.25:
#                 self.predict(dt)
#             self.has_measurement = False
#         return self.get_state(), near_edge

#     def get_state(self):
#         return self.x.flatten()  # [x, y, vx, vy]
    
#     def is_near_fov_edge(self, bbox2d_center_x, image_width, threshold=0.33):
#         """檢查目標是否接近影像邊緣"""
#         if bbox2d_center_x is None:
#             return False
#         edge_limit = image_width * threshold
#         if bbox2d_center_x < edge_limit or bbox2d_center_x > (image_width - edge_limit):
#             return True
#         return False


# class Yolo3DTransformNode:
#     def __init__(self):
#         self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(2.0))
#         self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

#         self.pub_pose = rospy.Publisher("transformed_bbox_center", Point, queue_size=10)
#         self.pub_trk = rospy.Publisher("trk3d_result", Trk3DArray, queue_size=10)
#         self.pub_marker = rospy.Publisher("trk3d_visual", MarkerArray, queue_size=10)

#         rospy.Subscriber("yolo_3d_result", Det3DArray, self.detection_callback)

#         self.trackers = {}  # {id: SimpleKalman}
#         self.last_time = None
#         self.duration = 0.1

#         # 可調參數
#         self.predict_only_timeout = 1.0  # 沒偵測時允許預測多久
#         self.remove_timeout = 2.5        # 超過多久沒更新就刪除

#     def detection_callback(self, msg):
#         trk3d_array = Trk3DArray()
#         trk3d_array.header = Header()
#         trk3d_array.header.frame_id = "odom"
#         trk3d_array.header.stamp = msg.header.stamp
#         marker_array = MarkerArray()

#         current_time = rospy.Time.now()
#         if self.last_time is None:
#             self.last_time = current_time
#             return
#         self.duration = (current_time - self.last_time).to_sec()
#         # if self.duration <= 0.04:
#         #     self.duration = 0.04
#         self.last_time = current_time

#         updated_ids = set()

#         # --- 更新有偵測到的ID ---
#         for det in msg.detections:
#             if det.tracked_id == -1:
#                 continue  # 跳過無效ID

#             bbox_center = det.center.position
#             bbox_center_stamped = PointStamped()
#             bbox_center_stamped.header = msg.header
#             bbox_center_stamped.point = Point(bbox_center.x, bbox_center.y, bbox_center.z)

#             try:
#                 bbox_center_odom = self.tf_buffer.transform(
#                     bbox_center_stamped, "odom", rospy.Duration(1.0)
#                 ).point

#                 obj_id = det.tracked_id
#                 if obj_id not in self.trackers:
#                     init_r = max(det.size.x, det.size.y) / 2
#                     self.trackers[obj_id] = SimpleKalman(bbox_center_odom.x, bbox_center_odom.y, init_r)

#                 kf = self.trackers[obj_id]
#                 state, near_edge = kf.step(self.duration, (bbox_center_odom.x, bbox_center_odom.y), det.bbox2d_center.x)
#                 x, y, vx, vy = state
#                 # rospy.loginfo("[3D Kalman] ID: %d | Pos: (%.2f, %.2f) | Vel: (%.2f, %.2f) | r: %.2f",
#                 #               obj_id, x, y, vx, vy, kf.radius)
#                 # rospy.loginfo("    BBox2D center in image: (%.2f, %.2f)", det.bbox2d_center.x, det.bbox2d_center.y)

#                 if not near_edge:
#                     r_new = max(det.size.x, det.size.y) / 2
#                     alpha = 0.3 if r_new > kf.radius else 0.1
#                     kf.radius = kf.radius * (1 - alpha) + r_new * alpha
#                     kf.radius = np.clip(kf.radius, 0.2, 0.45)

#                 updated_ids.add(obj_id)

#                 if kf.validity:
#                     self._publish_tracked_object(marker_array, trk3d_array, det, x, y, vx, vy)
#                 self.pub_pose.publish(bbox_center_odom)

#             except (tf2_ros.LookupException, tf2_ros.ExtrapolationException) as ex:
#                 rospy.logwarn_throttle(2.0, "TF transform failed: %s", ex)

#         # --- 沒偵測到的ID，繼續預測 ---
#         for obj_id, kf in list(self.trackers.items()):
#             if obj_id not in updated_ids:
#                 time_since_update = (current_time - kf.last_update).to_sec()
#                 if time_since_update < self.predict_only_timeout:
#                     state, _ = kf.step(self.duration, measurement=None)
#                     x, y, vx, vy = state
#                     if kf.validity:
#                         self._publish_predicted_object(marker_array, trk3d_array, obj_id, x, y, vx, vy)
#                 elif time_since_update > self.remove_timeout:
#                     del self.trackers[obj_id]

#         # --- 發布結果 ---
#         self.pub_trk.publish(trk3d_array)
#         self.pub_marker.publish(marker_array)

#     def _publish_tracked_object(self, marker_array, trk3d_array, det, x, y, vx, vy):
#         trk3d = Trk3D()
#         trk3d.tracked_id = det.tracked_id
#         trk3d.class_name = det.class_name
#         trk3d.x = x
#         trk3d.y = y
#         norm = math.hypot(vx, vy)
#         if norm < 0.25:
#             vx, vy = 0.0, 0.0
#         trk3d.vx = vx
#         trk3d.vy = vy
#         trk3d.yaw = math.atan2(vy, vx)
#         trk3d.radius = self.trackers[det.tracked_id].radius
#         trk3d.confidence = det.score
#         trk3d_array.trks_list.append(trk3d)

#         marker = self._make_marker(det.tracked_id, x, y, trk3d.yaw, trk3d.radius, [0.1, 0.9, 0.1])
#         marker_array.markers.append(marker)

#     def _publish_predicted_object(self, marker_array, trk3d_array, obj_id, x, y, vx, vy):
#         """沒有偵測更新時的預測結果（顏色不同）"""
#         trk3d = Trk3D()
#         trk3d.tracked_id = obj_id
#         trk3d.class_name = "predicted"
#         trk3d.x = x
#         trk3d.y = y
#         norm = math.hypot(vx, vy)
#         if norm < 0.25:
#             vx, vy = 0.0, 0.0
#         trk3d.vx = vx
#         trk3d.vy = vy
#         trk3d.yaw = math.atan2(vy, vx)
#         trk3d.radius = self.trackers[obj_id].radius
#         trk3d.confidence = 0.0
#         trk3d_array.trks_list.append(trk3d)

#         marker = self._make_marker(obj_id, x, y, trk3d.yaw, trk3d.radius, [0.9, 0.9, 0.1])
#         marker_array.markers.append(marker)

#     def _make_marker(self, obj_id, x, y, yaw, radius, rgb):
#         marker = Marker()
#         marker.header.frame_id = "odom"
#         marker.header.stamp = rospy.Time.now()
#         marker.ns = "people"
#         marker.id = obj_id
#         marker.type = Marker.CYLINDER
#         marker.action = Marker.ADD
#         marker.pose.position.x = x
#         marker.pose.position.y = y
#         marker.pose.position.z = 0.8
#         quat = tf_trans.quaternion_from_euler(0, 0, yaw)
#         marker.pose.orientation = Quaternion(*quat)
#         marker.scale.x = radius * 2.0
#         marker.scale.y = radius * 2.0
#         marker.scale.z = 1.6
#         marker.color.a = 0.6
#         marker.color.r = rgb[0]
#         marker.color.g = rgb[1]
#         marker.color.b = rgb[2]
#         marker.lifetime = rospy.Duration(1.0)
#         return marker


# if __name__ == '__main__':
#     rospy.init_node('yolo_3d_transform_node')
#     node = Yolo3DTransformNode()
#     rospy.spin()
