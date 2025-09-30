#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
SocialRuleSelector (rospy.Node)
├─ Subscribers:
│   ├─ /predict_trajs                        → callbackPredictTrajs
│   ├─ /move_base/GlobalPlan (nav_msgs/Path) → callbackGlobalPlan
│   ├─ /odom (nav_msgs/Odometry)             → callbackOdom
│   └─ /static_map (nav_msgs/OccupancyGrid)  → callbackMap
├─ Publisher:
│   └─ /social_rule (SocialRule)
└─ Main loop (10 Hz):
    1. checkCollision()
    2. if no collision → publish mode="normal"
    3. else:
       a. sampleTrajectories()
       b. classifyTrajectories()
       c. computeCosts()
       d. aggregateTypeCosts()
       e. selectBestType() → publish mode="yield", action_type=<best>
'''
import rospy
import math
import time
import numpy as np
from bisect import bisect_left
import cv2
import tf
import tf2_ros
import tf2_geometry_msgs
import tf.transformations as tft
from std_srvs.srv import Empty
from geometry_msgs.msg import PoseStamped, Quaternion, TwistStamped
from nav_msgs.msg import Path, OccupancyGrid, Odometry
from social_rules_selector.msg import PredictTrajs, SocialRule
from dynamic_reconfigure.client import Client
from visualization_msgs.msg import Marker, MarkerArray

class SocialRuleSelector:
    def __init__(self):
        # 1. Parameters
        self.rate = rospy.get_param('~rate', 10)
        self.safe_dist = rospy.get_param('~safe_dist', 0.5)
        self.sliding_time_window = rospy.get_param('~sliding_time_window', 1.0)  # seconds
        # Sampling parameters
        self.acc_samples = rospy.get_param('~acc_samples', 10)
        self.ang_samples = rospy.get_param('~ang_samples', 10)
        self.Ts = rospy.get_param('~sim_time', 1.0)
        self.v_min = rospy.get_param('~v_min', 0.0)  # m/s
        self.v_max = rospy.get_param('~v_max', 0.8)  # m/s
        # self.w_min = rospy.get_param('~w_min', -0.5263)  # rad/s
        self.w_max = rospy.get_param('~w_max', 0.5236)  # rad/s
        self.a_max = rospy.get_param('~a_max', 2.5)
        self.alpha_max = rospy.get_param('~alpha_max', 1.82)
        # cost weights : [obs, goal, path, speed, ped]
        self.weight_obs = rospy.get_param('~weight_obs', 10.0)
        self.weight_goal = rospy.get_param('~weight_goal', 1.0)
        self.weight_path = rospy.get_param('~weight_path', 0.5)
        self.weight_speed = rospy.get_param('~weight_speed', 0.1)
        self.weight_ped = rospy.get_param('~weight_ped', 10.0)
        rospy.loginfo(f"[SocialRuleSelector] Initialized weights: obs={self.weight_obs}, goal={self.weight_goal}, path={self.weight_path}, speed={self.weight_speed}, ped={self.weight_ped}")

        # TF 
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        # self.tf_listener = tf.TransformListener()
        self.map_frame = 'map'

        # Dynamic Reconfigure
        # self.dwa_client = Client('/move_base/DWAPlannerROS', timeout=10, config_callback=self.dyn_config_cb)
        self.teb_client = Client('/move_base/TebLocalPlannerROS', timeout=10, config_callback=self.dyn_config_cb)

        # 2. Subscribers
        self.sub_trajs = rospy.Subscriber('/predicted_trajectory', PredictTrajs, self.trajs_cb)
        # self.sub_path  = rospy.Subscriber('/move_base/NavfnROS/plan', Path, self.path_cb)
        self.sub_path  = rospy.Subscriber('/move_base/GlobalPlanner/plan', Path, self.path_cb)
        self.sub_odom  = rospy.Subscriber('/odom', Odometry, self.odom_cb)
        self.sub_map   = rospy.Subscriber('/map', OccupancyGrid, self.map_cb)
        # 3. Publisher
        self.pub_rule = rospy.Publisher('/social_rule', SocialRule, queue_size=1)
        self.pub_marker = rospy.Publisher('/social_rule_marker', Marker, queue_size=1)
        self.pub_sample_trajs = rospy.Publisher('/sampled_trajectories', MarkerArray, queue_size=1)
        # 4. Service
        self.clear_srv = rospy.ServiceProxy('/move_base/clear_costmaps', Empty)
        # 5. State
        self.person_trajs = None
        self.global_path = None
        self.current_odom = None
        self.map = None
        self.dist_field = None
        self.map_origin = (0.0, 0.0)
        self.map_info = None
        self.last_rule = None
        self.dist_field_saved = False

    # Callbacks
    def trajs_cb(self, msg):
        self.person_trajs = msg
    
    def path_cb(self, msg):
        self.global_path = msg

    def odom_cb(self, msg):
        self.current_odom = msg

    # Old version of map callback
    def map_cb_old(self, msg):
        self.map = msg
        # extract obstacle cell centers
        self.obstacle_positions = []
        width = msg.info.width
        height = msg.info.height
        res = msg.info.resolution
        ox = msg.info.origin.position.x
        oy = msg.info.origin.position.y
        data = msg.data
        for idx, val in enumerate(data):
            if val > 50:  # occupied
                x = ox + (idx % width + 0.5) * res
                y = oy + (idx // width + 0.5) * res
                self.obstacle_positions.append((x, y))

    # New version of map callback
    def map_cb(self, msg):
        # store map info and distance field
        self.map = msg
        self.map_info = msg.info
        w, h = msg.info.width, msg.info.height
        arr = np.array(msg.data, dtype=np.int8).reshape((h, w))

        # Save the raw map as an image only once
        if not hasattr(self, 'raw_map_saved') or not self.raw_map_saved:
            # Normalize the map data to 0-255 for visualization
            raw_map_image = np.zeros_like(arr, dtype=np.uint8)
            raw_map_image[arr == -1] = 127  # Unknown cells as gray
            raw_map_image[arr == 0] = 255   # Free cells as white
            raw_map_image[arr > 50] = 0    # Occupied cells as black
            cv2.imwrite('/home/andre/ros_docker_ws/catkin_ws/src/social_rules_selector/raw_map.png', raw_map_image)
            rospy.loginfo("Raw map saved as /home/andre/ros_docker_ws/catkin_ws/src/social_rules_selector/raw_map.png")
            self.raw_map_saved = True

        occ = ((arr > 50) | (arr == -1)).astype(np.uint8)  # Occupied cells and unknown cells treat as obstacles
        src = 1 - occ
        dist_px = cv2.distanceTransform(src, cv2.DIST_L2, 5)
        self.dist_field = dist_px * msg.info.resolution
        self.map_origin = (msg.info.origin.position.x, msg.info.origin.position.y)
        self.map_frame = msg.header.frame_id

        # Save the distance field as an image only once
        if not self.dist_field_saved:
            dist_image = (self.dist_field / self.dist_field.max() * 255).astype(np.uint8)  # Normalize to 0-255
            cv2.imwrite('/home/andre/ros_docker_ws/catkin_ws/src/social_rules_selector/distance_field.png', dist_image)  # Save the image
            rospy.loginfo("Distance field saved as /tmp/distance_field.png")
            cv2.imwrite('/home/andre/ros_docker_ws/catkin_ws/src/social_rules_selector/src_binary.png', src * 255)
            rospy.loginfo("Binary map saved as /home/andre/ros_docker_ws/catkin_ws/src/social_rules_selector/src_binary.png")
            self.dist_field_saved = True

    # Dynamic Reconfigure callback
    def dyn_config_cb(self, config):
        rospy.logdebug(f"Dynamic reconfigure called with config: {config}")

    def run(self):
        rate = rospy.Rate(self.rate)
        while not rospy.is_shutdown():
            missing = []
            if self.person_trajs is None:
                missing.append('predict_trajs')
            if self.global_path is None:
                missing.append('global_path')
            if self.current_odom is None:
                missing.append('odom')
            if self.dist_field is None:
                missing.append('distance_field')
            if missing:
                # rospy.logwarn("[SocialRuleSelector] Waiting for data: %s", missing)
                rate.sleep()
                continue

            # 1. checkCollision
            collision, collision_time_diff = self.checkCollision()
            if not collision:
                if self.last_rule != 'normal':
                    rospy.loginfo(f"[SocialRuleSelector] Selected rule: normal")
                self.v_max = 0.8  # m/s
                self.w_max = 0.5236  # rad/s
                self.teb_client.update_configuration({"max_vel_x": self.v_max, "max_vel_theta": self.w_max}) # theta approx 30 degrees
                self.last_rule = 'normal'
                self.publish_rule('normal')
                # self.publish_rule('turn_left')  # For testing, always publish turn_left
            else:
                # self.clear_srv()
                # 2. sampleTrajectories
                trajs = self.sampleTrajectories(collision_time_diff)
                # 3. classifyTrajectories
                classified = self.classifyTrajs(trajs)
                # 4. computeCosts
                costs = self.computeCosts(classified)
                # 5. aggregateTypeCosts
                avg_costs = self.aggregateTypeCosts(costs)
                # 6. selectBestType
                best = min(avg_costs, key=avg_costs.get)
                # best = 'turn_left' # For testing, always select turn_left
                # 7. dynamic reconfigure
                if best == 'accelerate' and self.last_rule != 'accelerate':
                    rospy.loginfo(f"[SocialRuleSelector] Selected rule: {best}")
                    self.v_max = 1.4  # m/s
                    self.w_max = 0.5236  # rad/s
                    self.teb_client.update_configuration({"max_vel_x": self.v_max, "max_vel_theta": self.w_max}) # theta approx 30 degrees
                elif best == 'decelerate' and self.last_rule != 'decelerate':
                    rospy.loginfo(f"[SocialRuleSelector] Selected rule: {best}")
                    self.v_max = 0.4  # m/s
                    self.w_max = 0.5236  # rad/s
                    self.teb_client.update_configuration({"max_vel_x": self.v_max, "max_vel_theta": self.w_max})
                elif best == 'turn_left' and self.last_rule != 'turn_left':
                    rospy.loginfo(f"[SocialRuleSelector] Selected rule: {best}")
                    self.v_max = 0.8  # m/s
                    self.w_max = 1.5708  # rad/s
                    self.teb_client.update_configuration({"max_vel_x": self.v_max, "max_vel_theta": self.w_max})
                elif best == 'turn_right' and self.last_rule != 'turn_right':
                    rospy.loginfo(f"[SocialRuleSelector] Selected rule: {best}")
                    self.v_max = 0.8  # m/s
                    self.w_max = 1.5708  # rad/s
                    self.teb_client.update_configuration({"max_vel_x": self.v_max, "max_vel_theta": self.w_max})
                # 8. publish
                self.last_rule = best
                self.publish_rule(best)
                self.publish_sampled_trajectories(trajs)
            rate.sleep()

    def checkCollision(self):   # Check if the robot will collide with any pedestrian in the next 5 seconds by trajectory from velocity prediction
        if self.person_trajs is None or self.current_odom is None:
            rospy.logwarn("[SocialRuleSelector] Missing person_trajs or current_odom, cannot check collision.")
            return False, None

        # === PARAMETERS ===
        dt = 0.1  # prediction time step in seconds
        steps = int(5.0 / dt)  # number of prediction steps
        r_robot = 0.25  # robot radius [m]
        r_person = 0.25  # person radius [m]
        collision_radius = r_robot + r_person

        # === ROBOT INITIAL STATE ===
        robot_pose = self.current_odom.pose.pose
        robot_twist = self.current_odom.twist.twist
        rx = robot_pose.position.x
        ry = robot_pose.position.y
        _, _, rtheta = tf.transformations.euler_from_quaternion([
            robot_pose.orientation.x,
            robot_pose.orientation.y,
            robot_pose.orientation.z,
            robot_pose.orientation.w
        ])
        vrx = robot_twist.linear.x * math.cos(rtheta)
        vry = robot_twist.linear.x * math.sin(rtheta)

        # === SIMULATE EACH TIME STEP ===
        for step in range(steps):
            t = step * dt
            # robot predicted position at time t
            rx_t = rx + vrx * t
            ry_t = ry + vry * t

            for pred_traj in self.person_trajs.predicted_trajs:
                if len(pred_traj.predicted_trajectory) <= step:
                    continue

                person_pose = pred_traj.predicted_trajectory[step].pose.position
                px_t = person_pose.x
                py_t = person_pose.y

                dist = math.sqrt((rx_t - px_t) ** 2 + (ry_t - py_t) ** 2)
                if dist <= collision_radius:
                    rospy.loginfo(f"[SocialRuleSelector] Predicted collision at t={t:.2f}s, distance={dist:.2f}m")
                    return True, t
        rospy.loginfo("[SocialRuleSelector] No collision detected in the predicted trajectory.")
        return False, None

    # def checkCollision(self):   # Check if the robot will collide with any pedestrian in the next sliding_time_window seconds by trajectory from global path
        # TODO: 按時間同步比對 global_path 與 predicted_trajs
        rospy.loginfo("[SocialRuleSelector] Checking for collisions...")
        # from odometry to get robot position and velocity
        ox = self.current_odom.pose.pose.position.x
        oy = self.current_odom.pose.pose.position.y
        vx = self.current_odom.twist.twist.linear.x
        vy = self.current_odom.twist.twist.linear.y
        v = math.hypot(vx, vy)
        if not hasattr(self, '_speed_info_logged'):
            self._speed_info_logged = False

        if v < 0.1:
            if not self._speed_info_logged:
                rospy.logdebug("Robot speed too small, using fallback v=0.1")
                self._speed_info_logged = True
            v = 0.1
        else:
            if self._speed_info_logged:
                rospy.logdebug("Robot speed is now greater than 0.1, stopping fallback v=0.1")
                self._speed_info_logged = False

        timed_path_odom = []
        t_accum_odom = self.person_trajs.header.stamp
        try:
            start_pose = PoseStamped()
            start_pose.header.stamp = self.person_trajs.header.stamp
            start_pose.header.frame_id = self.map_frame
            start_pose.pose.position.x = ox
            start_pose.pose.position.y = oy
            start_pose_odom = self.tf_buffer.transform(start_pose, 'odom', rospy.Duration(1.0))
            timed_path_odom.append((t_accum_odom, start_pose_odom.pose.position.x, start_pose_odom.pose.position.y))
        except (tf2_ros.TransformException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"Transform from {self.map_frame} to odom failed: {e}")
            exit(1)
        prev_x_odom, prev_y_odom = ox, oy
        for ps in self.global_path.poses:
            try:
                ps.header.stamp = self.person_trajs.header.stamp
                ps.header.frame_id = self.map_frame
                # Transform the global path point to the odom frame
                pose_odom = self.tf_buffer.transform(ps, 'odom', rospy.Duration(1.0))
                nx_odom = pose_odom.pose.position.x
                ny_odom = pose_odom.pose.position.y
            except (tf2_ros.TransformException, tf2_ros.ExtrapolationException) as e:
                rospy.logwarn(f"Transform from {self.map_frame} to odom failed: {e}")
                exit(1)
            dist = math.hypot(nx_odom - prev_x_odom, ny_odom - prev_y_odom)
            dt = rospy.Duration(dist / v)
            t_accum_odom += dt
            timed_path_odom.append((t_accum_odom, nx_odom, ny_odom))
            prev_x_odom, prev_y_odom = nx_odom, ny_odom

        timed_path_sec = [t[0].to_sec() for t in timed_path_odom]

        # For each predicted trajectory, check if it intersects with the global path
        for traj in self.person_trajs.predicted_trajs:
            # Check if the trajectory is empty
            if not traj.predicted_trajectory:
                rospy.logwarn("Empty predicted trajectory, skipping Group ID: %d", traj.group_id)
                continue

            pvx = traj.predicted_velocity.linear.x
            pvy = traj.predicted_velocity.linear.y

            for person_traj_point in traj.predicted_trajectory:
                t_pred = person_traj_point.header.stamp
                idx = bisect_left(timed_path_sec, t_pred.to_sec() - self.sliding_time_window)
                px = person_traj_point.pose.position.x
                py = person_traj_point.pose.position.y
                while idx < len(timed_path_odom) and timed_path_sec[idx] < t_pred.to_sec() + self.sliding_time_window:
                    t_i, x_odom, y_odom = timed_path_odom[idx]
                    idx += 1
                    # Check if the predicted point is within the safe distance from the global path
                    dx = px - x_odom
                    dy = py - y_odom
                    # TODO: dynamically adjust the safe distance based on the importance (need to apply in tracked groups) of the pedestrian
                    importance = 1.0
                    if math.hypot(dx, dy) < self.safe_dist * importance + v / self.a_max: 
                        collision_time_diff = t_i - timed_path_odom[0][0]
                        # print out the collision info about where robot is and where the pedestrian is and where the predict robot is and when collision happens
                        rospy.loginfo(f"Collision detected! Now Time: {rospy.Time.now().to_sec()}")
                        rospy.loginfo(f"Robot Position: ({ox}, {oy}), Velocity: ({vx}, {vy})")
                        # rospy.loginfo(f"Pedestrian Group Position: ({traj.predicted_center.pose.position.x}, {traj.predicted_center.pose.position.y}), Velocity: ({pvx}, {pvy})")
                        rospy.loginfo(f"Predicted Pedestrian Position: ({px}, {py}), Time: {t_pred.to_sec()}")
                        rospy.loginfo(f"Global Path Point: ({x_odom}, {y_odom}), Time: {t_i.to_sec()}")
                        return True, collision_time_diff.to_sec()
                        # TODO: Consider both distance and velocity
                        # # Relative velocity (pedestrian - robot)
                        # rel_vx = pvx - vx
                        # rel_vy = pvy - vy
                        # # Check if the relative velocity is in the same direction as the relative position
                        # if rel_vx * dx + rel_vy * dy < 0:
                        #     rospy.loginfo(f"Collision detected! Group ID: {traj.group_id}, Time: {t_pred.to_sec()}, Position: ({px}, {py})")
                        #     rospy.loginfo(f"Robot Now Position: ({ox}, {oy}), Velocity: ({vx}, {vy})")
                        #     rospy.loginfo(f"Global Path Point: ({x_odom}, {y_odom})")
                        #     return True
        return False, None

    # def checkCollision(self):   # Check if the robot will collide with any pedestrian in the next 5 seconds by trajectory from global path
        # 取得機器人當前位置與速度
        ox = self.current_odom.pose.pose.position.x
        oy = self.current_odom.pose.pose.position.y
        vx = self.current_odom.twist.twist.linear.x
        vy = self.current_odom.twist.twist.linear.y
        v = math.hypot(vx, vy)

        if not hasattr(self, '_speed_info_logged'):
            self._speed_info_logged = False

        if v < 0.1:
            if not self._speed_info_logged:
                rospy.logdebug("Robot speed too small, using fallback v=0.1")
                self._speed_info_logged = True
            v = 0.1
        else:
            if self._speed_info_logged:
                rospy.logdebug("Robot speed is now greater than 0.1, stopping fallback v=0.1")
                self._speed_info_logged = False

        # 轉換起始點到 odom frame
        try:
            start_pose = PoseStamped()
            start_pose.header.stamp = self.person_trajs.header.stamp
            start_pose.header.frame_id = self.map_frame
            start_pose.pose.position.x = ox
            start_pose.pose.position.y = oy
            start_pose_odom = self.tf_buffer.transform(start_pose, 'odom', rospy.Duration(1.0))
            start_x = start_pose_odom.pose.position.x
            start_y = start_pose_odom.pose.position.y
        except (tf2_ros.TransformException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"Transform from {self.map_frame} to odom failed: {e}")
            exit(1)

        # 生成 5 秒內的 global path 座標（轉換到 odom frame）
        timed_path_odom = [(0.0, start_x, start_y)]
        prev_x_odom, prev_y_odom = start_x, start_y
        accum_time = 0.0  # 累計時間

        for ps in self.global_path.poses:
            try:
                ps.header.stamp = self.person_trajs.header.stamp
                ps.header.frame_id = self.map_frame
                pose_odom = self.tf_buffer.transform(ps, 'odom', rospy.Duration(1.0))
                nx_odom = pose_odom.pose.position.x
                ny_odom = pose_odom.pose.position.y
            except (tf2_ros.TransformException, tf2_ros.ExtrapolationException) as e:
                rospy.logwarn(f"Transform from {self.map_frame} to odom failed: {e}")
                exit(1)

            dist = math.hypot(nx_odom - prev_x_odom, ny_odom - prev_y_odom)
            dt = dist / v
            accum_time += dt
            if accum_time > 5.0:
                break
            timed_path_odom.append((accum_time, nx_odom, ny_odom))
            prev_x_odom = nx_odom
            prev_y_odom = ny_odom

        # 對每一組行人的預測軌跡進行交會檢查
        for traj in self.person_trajs.predicted_trajs:
            if not traj.predicted_trajectory:
                rospy.logwarn("Empty predicted trajectory, skipping Group ID: %d", traj.group_id)
                continue

            pvx = traj.predicted_velocity.linear.x
            pvy = traj.predicted_velocity.linear.y

            for person_traj_point in traj.predicted_trajectory:
                px = person_traj_point.pose.position.x
                py = person_traj_point.pose.position.y
                t_pred = person_traj_point.header.stamp

                for accum_time_odom, x_odom, y_odom in timed_path_odom:
                    dx = px - x_odom
                    dy = py - y_odom
                    importance = 1.0  # TODO: 可根據 group importance 調整
                    if math.hypot(dx, dy) < self.safe_dist * importance + v / self.a_max:
                        rospy.loginfo("Collision detected!")
                        rospy.loginfo(f"Robot Position: ({ox}, {oy}), Velocity: ({vx}, {vy})")
                        rospy.loginfo(f"Pedestrian Group Position: ({traj.predicted_center.pose.position.x}, {traj.predicted_center.pose.position.y}), Velocity: ({pvx}, {pvy})")
                        rospy.loginfo(f"Predicted Pedestrian Position: ({px}, {py}), Time: {t_pred.to_sec()}")
                        rospy.loginfo(f"Global Path Point: ({x_odom}, {y_odom})")
                        return True, accum_time_odom

        return False, None

    def sampleTrajectories(self, collision_time_diff=None):
        rospy.loginfo("Sampling trajectories...")
        trajs = []
        # from odometry to get robot position
        ox = self.current_odom.pose.pose.position.x
        oy = self.current_odom.pose.pose.position.y
        # from odometry to get robot orientation
        quaternion = self.current_odom.pose.pose.orientation
        _, _, theta = tft.euler_from_quaternion([
            quaternion.x, quaternion.y, quaternion.z, quaternion.w
        ])
        v0 = self.current_odom.twist.twist.linear.x # Because robot is differential drive, only x is used
        w0 = self.current_odom.twist.twist.angular.z # In radians

        if collision_time_diff is not None and collision_time_diff > 3.0:
            collision_time_diff = 3.0  # Limit the maximum collision time to 3 seconds

        # Calculate dynamic window
        v_min = max(self.v_min, v0 - self.a_max * (collision_time_diff if collision_time_diff is not None and collision_time_diff > self.Ts else self.Ts))
        v_max = min(1.4, v0 + self.a_max * (collision_time_diff if collision_time_diff is not None and collision_time_diff > self.Ts else self.Ts))
        w_min = max(-1.5708, w0 - self.alpha_max * (collision_time_diff if collision_time_diff is not None and collision_time_diff > self.Ts else self.Ts))
        w_max = min(1.5708, w0 + self.alpha_max * (collision_time_diff if collision_time_diff is not None and collision_time_diff > self.Ts else self.Ts))

        dt = 0.2  # Simulation time step
        steps = int(collision_time_diff / dt) if collision_time_diff is not None and collision_time_diff > self.Ts else int(self.Ts / dt)

        for i in range(self.acc_samples):
            v = v_min + (v_max - v_min) * i / max(self.acc_samples - 1, 1)
            for j in range(self.ang_samples):
                w = w_min + (w_max - w_min) * j / max(self.ang_samples - 1, 1)
                traj = []
                x, y, th = ox, oy, theta
                current_time = rospy.Time.now()
                for k in range(steps):
                    # forward simulate
                    x += v * dt * math.cos(th)
                    y += v * dt * math.sin(th)
                    th += w * dt
                    ps = PoseStamped()
                    ps.header.stamp = current_time + rospy.Duration(dt * (k + 1))
                    ps.header.frame_id = 'odom'
                    ps.pose.position.x = x
                    ps.pose.position.y = y
                    q = tft.quaternion_from_euler(0, 0, th)
                    ps.pose.orientation = Quaternion(*q)
                    traj.append(ps)
                trajs.append(traj)
        return trajs

    def classifyTrajs(self, trajs):
        rospy.loginfo("Classifying trajectories...")
        # Classify trajectories based on their end orientation and speed
        classified = {'accelerate': [], 'decelerate': [], 'turn_right': [], 'turn_left': []}
        v0 = self.current_odom.twist.twist.linear.x
        angle_thresh = math.radians(30)
        for traj in trajs:
            if not traj:
                continue
            # Initial and final orientation
            q0 = traj[0].pose.orientation
            _, _, th0 = tft.euler_from_quaternion([q0.x, q0.y, q0.z, q0.w])
            q1 = traj[-1].pose.orientation
            _, _, th1 = tft.euler_from_quaternion([q1.x, q1.y, q1.z, q1.w])
            dth = (th1 - th0 + math.pi) % (2*math.pi) - math.pi  # Normalize to [-pi, pi]
            # If oreientation is large than the threshold, classify as turn
            if abs(dth) > angle_thresh:
                if dth > 0:
                    classified['turn_left'].append(traj)
                else:
                    classified['turn_right'].append(traj)
            else:
                # Calculate the mean velocity
                dt = (traj[-1].header.stamp - traj[0].header.stamp).to_sec()
                dx = traj[-1].pose.position.x - traj[0].pose.position.x
                dy = traj[-1].pose.position.y - traj[0].pose.position.y
                vend = math.hypot(dx, dy) / dt if dt > 1e-3 else 0.0
                if vend > v0:
                    classified['accelerate'].append(traj)
                else:
                    classified['decelerate'].append(traj)
        return classified

    def computeCosts(self, classified):
        rospy.loginfo("Computing costs for classified trajectories...")
        costs = {}
        # global path points (Map frame)
        path_pts = [(p.pose.position.x, p.pose.position.y) for p in self.global_path.poses]

        # Measure how much time it takes to lookup transform from odom to map
        start_time = time.time()
        try:
            trans = self.tf_buffer.lookup_transform(self.map_frame, 'odom', rospy.Time(0), rospy.Duration(1.0))
        except (tf2_ros.TransformException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"Transform from odom to map failed: {e}")
            return
        elapsed_time = time.time() - start_time
        rospy.loginfo(f"Lookup transform from odom to map took {elapsed_time:.5f} seconds")

        t = trans.transform.translation
        trans_vec = np.array([t.x, t.y, t.z])
        q = trans.transform.rotation
        rot_mat = tft.quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]

        # Measure how much time it takes to transform pedestrian trajectories to map frame
        start_time = time.time() 
        # predicted traj points (transform to map frame)
        ped_pts = []
        # for traj in self.person_trajs.predicted_trajs:
        #     for p in traj.predicted_trajectory:
        #         ps_map = tf2_geometry_msgs.do_transform_pose(p, trans)
        #         ped_pts.append((ps_map.pose.position.x, ps_map.pose.position.y))
        ped_points_np = np.array([[p.pose.position.x, p.pose.position.y, p.pose.position.z]
                      for traj in self.person_trajs.predicted_trajs
                      for p in traj.predicted_trajectory])
        # Transform points to map frame
        ped_points_map = (rot_mat @ ped_points_np.T).T + trans_vec
        ped_pts = ped_points_map[:, :2]
        elapsed_time = time.time() - start_time
        rospy.loginfo(f"Transformed {len(ped_pts)} pedestrian trajectory points to map frame in {elapsed_time:.5f} seconds")

        # Measure how much time it takes to transform classified trajectories to map frame
        start_time = time.time()
        mapped_trajs = {}
        for typ, trajs in classified.items():
            mapped_trajs[typ] = []
            for traj in trajs:
                # traj_map = [tf2_geometry_msgs.do_transform_pose(ps, trans) for ps in traj]
                # mapped_trajs[typ].append(traj_map)
                traj_points_np = np.array([
                    [ps.pose.position.x, ps.pose.position.y, ps.pose.position.z]
                    for ps in traj
                ])
                traj_points_map = (rot_mat @ traj_points_np.T).T + trans_vec
                mapped_trajs[typ].append(traj_points_map)
        elapsed_time = time.time() - start_time
        rospy.loginfo(f"Transformed {len(classified)} classified trajectories to map frame in {elapsed_time:.5f} seconds")

        # Measure how much time it takes to compute costs
        start_time = time.time()
        path_pts_np = np.array(path_pts)  # (P, 2)
        ped_pts_np = ped_pts if ped_pts.size > 0 else np.empty((0, 2))  # (K, 2)
        for typ, trajs in mapped_trajs.items():
            costs[typ] = []
            for traj_map in trajs:
                traj_np = traj_map[:, :2]  # 取 XY  # (T, 2)

                # -----------------------------
                # Obs cost
                # -----------------------------
                ix = np.clip(((traj_np[:, 0] - self.map_origin[0]) / self.map_info.resolution).astype(int),
                            0, self.map_info.width - 1)
                iy = np.clip(((traj_np[:, 1] - self.map_origin[1]) / self.map_info.resolution).astype(int),
                            0, self.map_info.height - 1)
                obs_vals = 1.0 / (self.dist_field[iy, ix] + 1e-3)
                c_obs = np.max(obs_vals) if obs_vals.size > 0 else 0.0

                # -----------------------------
                # Goal cost
                # -----------------------------
                end_x, end_y = traj_np[-1]
                N = min(len(self.global_path.poses), 100)
                subgoal = self.global_path.poses[N-1].pose.position
                c_goal = math.hypot(end_x - subgoal.x, end_y - subgoal.y)

                # -----------------------------
                # Path diff cost
                # -----------------------------
                if path_pts_np.size > 0:
                    dist_matrix = np.linalg.norm(traj_np[:, None, :] - path_pts_np[None, :, :], axis=2)
                    path_diffs = np.min(dist_matrix, axis=1)
                    c_path = np.mean(path_diffs)
                else:
                    c_path = 0.0

                # -----------------------------
                # Ped cost
                # -----------------------------
                if ped_pts_np.size > 0:
                    dist_matrix = np.linalg.norm(traj_np[:, None, :] - ped_pts_np[None, :, :], axis=2)
                    ped_vals = 1.0 / (np.min(dist_matrix, axis=1) + 1e-3)
                    c_ped = np.max(ped_vals)
                else:
                    c_ped = 0.0

                # -----------------------------
                # Total cost
                # -----------------------------
                total = (self.weight_obs * c_obs +
                        self.weight_goal * c_goal +
                        self.weight_path * c_path +
                        self.weight_ped * c_ped)
                costs[typ].append(total)
                rospy.loginfo(f"Type: {typ}, Cost: {total:.4f}, Obs: {c_obs:.4f}, Goal: {c_goal:.4f}, Path: {c_path:.4f}, Ped: {c_ped:.4f}")
        elapsed_time = time.time() - start_time
        rospy.loginfo(f"Computed costs for {len(classified)} classified types in {elapsed_time:.5f} seconds")
        return costs

    def aggregateTypeCosts(self, costs, lower_q=5, upper_q=95):
        rospy.loginfo("Aggregating costs for each type...")
        # if not costs:
        #     rospy.logwarn("No costs to aggregate, returning empty dictionary")
        #     return {}
        agg = {}
        all_vals = [v for lst in costs.values() for v in lst if lst]
        global_min = min(all_vals)
        global_max = max(all_vals)

        q_low = np.percentile(all_vals, lower_q)
        q_high = np.percentile(all_vals, upper_q)
        rospy.loginfo(f"Using {lower_q}%={q_low:.3f}, {upper_q}%={q_high:.3f} for normalization")
        scale = q_high - q_low if q_high > q_low else 1e-6

        means = [np.mean(lst) for lst in costs.values() if lst]
        ratio = max(means) / min(means)
        print(f"Mean ratio = {ratio:.2f}")

        for typ, lst in costs.items():
            if not lst:
                rospy.logwarn(f"Empty cost list for {typ}, setting to inf")
                costs[typ] = None
                continue

            mean = np.mean(lst)
            std = np.std(lst)
            minv = np.min(lst)
            maxv = np.max(lst)
            print(f"{typ}: mean={mean:.3f}, std={std:.3f}, range=({minv:.3f}, {maxv:.3f})")

            # min-max normalization
            # norm = [(c - global_min) / (global_max - global_min + 1e-6) for c in lst]
            # agg[typ] = sum(norm) / len(norm)
            # robust min-max normalization
            norm = [(min(max(c, q_low), q_high) - q_low) / scale for c in lst]
            agg[typ] = sum(norm) / len(norm)
            rospy.loginfo(f"Costs for {typ}: {agg[typ]} ; # of trajs: {len(lst)}")
        return agg

    def publish_rule(self, rule):
        msg = SocialRule(rule=rule)
        self.pub_rule.publish(msg)
        self.publish_rule_marker(rule)

    def publish_rule_marker(self, rule):
        marker = Marker()
        marker.header.frame_id = "base_link"  
        marker.header.stamp = rospy.Time.now()
        marker.ns = "rule"
        marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = 0.0  # 調整顯示位置（相對 base_link）
        marker.pose.position.y = 0.0
        marker.pose.position.z = 1.0
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.5  # 文字大小
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.text = f"Rule: {rule}"
        self.pub_marker.publish(marker)

    def publish_sampled_trajectories(self, trajs):
        # self.clear_sampled_trajectories()  # 清掉上一輪 marker
        marker_array = MarkerArray()
        for i, traj in enumerate(trajs):
            marker = Marker()
            marker.header.frame_id = "odom"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "sample_trajs"
            marker.id = i
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.02  # 線寬
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.lifetime = rospy.Duration(0.5)  # 持續時間

            for pose in traj:
                p = pose.pose.position
                marker.points.append(p)

            marker_array.markers.append(marker)

        self.pub_sample_trajs.publish(marker_array)

    def clear_sampled_trajectories(self):
        marker = Marker()
        marker.action = Marker.DELETEALL
        marker.header.frame_id = "odom"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "sample_trajs"
        self.pub_sample_trajs.publish(MarkerArray(markers=[marker]))

if __name__=='__main__':
    rospy.init_node('social_rule_selector')
    node = SocialRuleSelector()
    node.run()
