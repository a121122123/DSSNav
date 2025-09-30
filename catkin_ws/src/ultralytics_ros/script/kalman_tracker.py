# kalman_tracker.py
import numpy as np
from filterpy.kalman import KalmanFilter
import time
import rospy

class Tracker:
    def __init__(self, object_id, class_name, confidence, x, y, r, dt=1, sigma=1.0, q=0.01):
        self.object_id = object_id
        self.class_name = class_name
        self.confidence = confidence
        self.r = r
        self.dt = dt
        self.age = 0  # Track age for deletion
        self.validity = False
        self.last_seen = rospy.Time.now()
        self.last_state = (x, y)
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        
        # State transition matrix
        self.kf.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Measurement matrix
        # We only measure the position (x, y), not the velocity
        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # Process noise covariance
        # q is the process noise variance, which can be adjusted based on the expected motion dynamics
        self.kf.Q = q * np.eye(4)

        # Measurement noise covariance
        # sigma is the measurement noise standard deviation, which can be adjusted based on sensor accuracy
        self.kf.R = sigma**2 * np.eye(2)

        # Initial state covariance
        # We assume a large initial uncertainty in the state
        # P is the state covariance matrix, which represents the uncertainty in the state estimate
        # We set the diagonal elements to large values to indicate high uncertainty in position and velocity
        self.kf.P = np.diag([10, 10, 100, 100])

        # Initial state
        self.kf.x = np.array([x, y, 0, 0])

    def update(self, x, y, dt):
        # self.dt = dt
        # # rospy.loginfo("dt: %.3f", dt)
        # # Update the Kalman filter with the new position measurement
        # # The Kalman filter will predict the next state based on the previous state and the time step,
        # # and then update the state with the new measurement
        # self.kf.F[0, 2] = dt
        # self.kf.F[1, 3] = dt

        # # 更新 Q 矩陣
        # sigma_a = 0.5  # 加速度雜訊參數（根據實驗可調）
        # q = sigma_a ** 2
        # self.kf.Q = np.array([
        #     [q*dt**4/4, 0, q*dt**3/2, 0],
        #     [0, q*dt**4/4, 0, q*dt**3/2],
        #     [q*dt**3/2, 0, q*dt**2, 0],
        #     [0, q*dt**3/2, 0, q*dt**2]
        # ])

        self.kf.predict()
        self.kf.update(np.array([x, y]))
        self.last_seen = rospy.Time.now()
        self.age += 1
        if self.age > 5:
            self.validity = True

    def predict(self, dt):
        self.dt = dt
        # rospy.loginfo("dt: %.3f", dt)
        # Update the Kalman filter with the new position measurement
        # The Kalman filter will predict the next state based on the previous state and the time step,
        # and then update the state with the new measurement
        self.kf.F[0, 2] = dt
        self.kf.F[1, 3] = dt

        # 更新 Q 矩陣
        sigma_a = 0.5  # 加速度雜訊參數（根據實驗可調）
        q = sigma_a ** 2
        self.kf.Q = np.array([
            [q*dt**4/4, 0, q*dt**3/2, 0],
            [0, q*dt**4/4, 0, q*dt**3/2],
            [q*dt**3/2, 0, q*dt**2, 0],
            [0, q*dt**3/2, 0, q*dt**2]
        ])
        # Predict the next state based on the current state and the time step
        self.kf.predict()

    def get_state(self):
        return self.kf.x

class MultiObjectTracker:
    def __init__(self):
        self.trackers = {}

    def add_object(self, object_id, class_name, confidence, x, y, r):
        if object_id not in self.trackers:
            self.trackers[object_id] = Tracker(object_id, class_name, confidence, x, y, max(r, 0.25))  # Ensure radius is at least 0.25 to avoid division by zero
            # rospy.loginfo("Add object ID: %d | Class: %s | Confidence: %.2f | Pos: (%.2f, %.2f) | r: %.2f",
            #             object_id, class_name, confidence, x, y, r)
        else:
            self.trackers[object_id].kf.x = np.array([x, y, 0, 0])
            self.trackers[object_id].class_name = class_name
            self.trackers[object_id].confidence = confidence
            self.trackers[object_id].r = r
            # rospy.loginfo("Update non-tracked object ID: %d | Pos: (%.2f, %.2f)", object_id, x, y)

    def update_object(self, object_id, class_name, confidence, x, y, r, dt):
        if object_id in self.trackers:
            tracker = self.trackers[object_id]
            # TODO:驗證一下要用偵測到的位置差，還是用Kalman內的位置
            # last_state = tracker.get_state()
            # last_x, last_y = last_state[0], last_state[1]
            last_x, last_y = tracker.last_state

            dist = np.hypot(x - last_x, y - last_y)
            if dist > 0.2: # 正常人跑步速度/scan頻率
                # tracker.kf.predict()  # 只做預測，不更新
                # rospy.logwarn(f"[Kalman] Rejected update for object {object_id}: distance jump too large ({dist:.2f} m)")
                # return  # 跳過這次更新，視為異常值
                rospy.logdebug(f"[Kalman] Update for object {object_id}: distance jump too large ({dist:.2f} m). Using last position.")
                rospy.logdebug(f"    Last pos: ({last_x:.2f}, {last_y:.2f}), New pos: ({x:.2f}, {y:.2f})")
                tracker.update(last_x, last_y, dt)
            else:
                tracker.update(x, y, dt)
            tracker.class_name = class_name
            tracker.confidence = confidence
            tracker.r = max(tracker.r, r)
            tracker.last_state = (x, y)
        else:
            self.add_object(object_id, class_name, confidence, x, y, r)

    def predict_object(self, dt):
        # Predict the next state for all tracked objects
        for obj_id in self.trackers:
            self.trackers[obj_id].predict(dt)

    def remove_old_objects(self, timeout=2.0):
        current_time = rospy.Time.now()
        # Remove objects that have not been seen for a specified timeout
        to_remove = [obj_id for obj_id, tracker in self.trackers.items() if (current_time - tracker.last_seen).to_sec() > timeout]

        for obj_id in to_remove:
            del self.trackers[obj_id]
            # rospy.loginfo("Remove object ID: %d", obj_id)
