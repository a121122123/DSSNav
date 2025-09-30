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
from geometry_msgs.msg import PoseStamped, Quaternion, TwistStamped, Point
from nav_msgs.msg import Path, OccupancyGrid, Odometry
from social_rules_selector.msg import PredictTrajs, SocialRule, TrackedGroups
from dynamic_reconfigure.client import Client
from visualization_msgs.msg import Marker, MarkerArray

class SocialRuleSelector:
    def __init__(self):
        # 1. Parameters
        self.rate = rospy.get_param('~rate', 10)
        self.safe_dist = rospy.get_param('~safe_dist', 0.5)
        self.always_static = rospy.get_param('~always_static', False)  # 是否固定為 normal
        self.local_planner = rospy.get_param('~local_planner', 'teb')  # 'teb' 或 'dwa'
        rospy.loginfo(f"[SocialRuleSelector] Using local_planner: {self.local_planner}, always_static: {self.always_static}")

        # Sampling parameters
        self.v_max = rospy.get_param('~v_max', 0.7)  # m/s
        self.w_max = rospy.get_param('~w_max', 1.0)  # rad/s
        self.a_max = rospy.get_param('~a_max', 2.5)

        # cost weights : [progress, similarity, proximity]
        self.weight_progress = rospy.get_param('~weight_progress', -1.0)
        self.weight_similarity = rospy.get_param('~weight_similarity', 1.0)
        self.weight_proximity = rospy.get_param('~weight_proximity', 7.0)
        rospy.loginfo(f"[SocialRuleSelector] Initialized weights: progress={self.weight_progress}, similarity={self.weight_similarity}, proximity={self.weight_proximity}")

        # Rule activation
        self.rule_activation = {
            "normal": 1.0,
            "accelerate": 1.0,
            "decelerate": 1.0,
            "turn_left": 1.0,
            "turn_right": 1.0,
        }
        self.activation_increase = 0.1   # 被選中的增加量
        self.activation_decay = 0.05     # 沒被選中的衰減量
        self.rule_activation_min = rospy.get_param('~rule_activation_min', 1.0)  # 最低值
        self.rule_activation_max = rospy.get_param('~rule_activation_max', 2.0)  # 最高值

        # TF 
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.map_frame = 'map'

        # Dynamic Reconfigure clients
        if self.local_planner == 'teb':
            self.planner_client = Client('/move_base/TebLocalPlannerROS', timeout=10, config_callback=self.dyn_config_cb)
        elif self.local_planner == 'dwa':
            self.planner_client = Client('/move_base/DWAPlannerROS', timeout=10, config_callback=self.dyn_config_cb)
        else:
            rospy.logwarn(f"[SocialRuleSelector] Unknown local_planner: {self.local_planner}, defaulting to TEB")
            self.planner_client = Client('/move_base/TebLocalPlannerROS', timeout=10, config_callback=self.dyn_config_cb)

        # 2. Subscribers
        self.sub_trajs = rospy.Subscriber('/predicted_trajectory', PredictTrajs, self.trajs_cb)
        self.sub_groups = rospy.Subscriber('/tracked_groups', TrackedGroups, self.groups_cb)
        # self.sub_path  = rospy.Subscriber('/move_base/NavfnROS/plan', Path, self.path_cb)
        self.sub_path  = rospy.Subscriber('/move_base/GlobalPlanner/plan', Path, self.path_cb)
        self.sub_odom  = rospy.Subscriber('/odom', Odometry, self.odom_cb)
        self.sub_map   = rospy.Subscriber('/map', OccupancyGrid, self.map_cb)

        # 3. Publisher
        self.pub_rule = rospy.Publisher('/social_rule', SocialRule, queue_size=1)
        self.pub_marker = rospy.Publisher('/social_rule_marker', Marker, queue_size=1)
        self.pub_sample_trajs = rospy.Publisher('/sampled_trajectories', MarkerArray, queue_size=1)

        # 4. State
        self.person_trajs = None
        self.groups_state = None
        self.predicted_groups = []
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

    def groups_cb(self, msg):
        self.groups_state = msg

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
            if self.groups_state is None:
                missing.append('tracked_groups')
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

            if self.always_static:
                best = "normal"
                trajs = {}
            else:
                # 1. checkCollision
                is_collision, collisions = self.checkCollision(
                    np.array([self.current_odom.pose.pose.position.x, self.current_odom.pose.pose.position.y]),
                    np.array([self.current_odom.twist.twist.linear.x, self.current_odom.twist.twist.linear.y]),
                    self.groups_state.groups
                )

                if is_collision:
                    rospy.logdebug(f"[SocialRuleSelector] Collision detected")
                    rospy.logdebug(f"[SocialRuleSelector] Collision details: {collisions}")

                    # 2. sampleTrajectories
                    trajs = self.sampleTrajectories(collisions)
                    # 3. removeInvalidTrajs
                    trajs = self.removeInvalidTrajs(trajs)
                    # 4. computeCosts
                    costs = self.computeCosts(trajs)
                    # 5. select candidate rule by cost
                    best_raw = min(costs, key=costs.get)
                else:
                    # 沒有碰撞 → 候選就是 normal
                    best_raw = "normal"
                    trajs = {}

                # 6. update activation
                self.rule_activation[best_raw] = min(self.rule_activation_max,
                                                    self.rule_activation[best_raw] + self.activation_increase)
                for rule in self.rule_activation:
                    if rule != best_raw:
                        self.rule_activation[rule] = max(self.rule_activation_min,
                                                        self.rule_activation[rule] - self.activation_decay)
                # 7. select final best by activation
                best = max(self.rule_activation, key=self.rule_activation.get)

                # --- Debug log: 印出每個 rule 的 activation ---
                activation_str = ", ".join([f"{r}: {a:.2f}" for r, a in self.rule_activation.items()])
                rospy.logdebug(f"[SocialRuleSelector] Activation values -> {activation_str}, Selected: {best}")

            # 8. dynamic reconfigure (只在切換時更新參數)
            # --- 根據 local_planner 切換 ---
            if best != self.last_rule:
                rospy.loginfo(f"[SocialRuleSelector] Switching to rule: {best}")

                if best == 'normal':
                    self.v_max, self.w_max = 0.7, 1.0
                elif best == 'accelerate':
                    self.v_max, self.w_max = 1.0, 1.0
                elif best == 'decelerate':
                    self.v_max, self.w_max = 0.4, 1.0
                elif best in ['turn_left', 'turn_right']:
                    self.v_max, self.w_max = 0.7, 1.5708

                self.planner_client.update_configuration({
                    "max_vel_x": self.v_max,
                    "max_vel_theta": self.w_max
                })

            # 9. publish
            self.last_rule = best
            self.publish_rule(best)
            if trajs:
                self.publish_sampled_trajectories(trajs)

            rate.sleep()

    def checkCollision(self, p_r, v_r, groups, epsilon=0.8, min_speed_threshold=0.05):
        """
        預測是否會與人 j 發生潛在碰撞

        Parameters:
            p_r: np.array, 機器人位置 (2D)
            v_r: np.array, 機器人速度 (2D)
            groups: list, 人群列表，每個元素包含 group_id, centerOfGravity, group_velocity
            epsilon: float, bearing angle 的閾值（弧度）

        Returns:
            is_collision: bool, 是否有潛在碰撞
            collisions: list, 每個潛在碰撞的詳細資訊，包括 group_id, TTC, phi, future_pos, is_collision
        """
        # # rospy.loginfo(f"[SocialRuleSelector] Checking collision with robot at {p_r} with velocity {v_r}")

        # if np.linalg.norm(v_r) < min_speed_threshold:
        #     v_r = np.zeros_like(v_r)
        #     return False, []

        # collisions = []

        # for group in groups:
        #     # rospy.loginfo(f"[SocialRuleSelector] Checking collision with group {group.group_id}")
        #     p_j = np.array([group.centerOfGravity.pose.position.x, group.centerOfGravity.pose.position.y])
        #     v_j = np.array([group.group_velocity.linear.x, group.group_velocity.linear.y])

        #     d = p_j - p_r                     # 相對位置
        #     v_rel = v_j - v_r                # 相對速度

        #     d_norm = np.linalg.norm(d)
        #     v_rel_norm = np.linalg.norm(v_rel)

        #     rospy.logdebug(f"[SocialRuleSelector] Group {group.group_id}: d={d}, ||d||={d_norm}, v_rel={v_rel}, ||v_rel||={v_rel_norm}")

        #     if d_norm == 0 or v_rel_norm == 0:
        #         rospy.logwarn(f"[SocialRuleSelector] Skipping group {group.group_id} due to zero distance {d_norm} or relative speed {v_rel_norm}.")
        #         continue

        #     # Bearing angle (夾角)
        #     cos_phi = np.clip(-np.dot(d, v_rel) / (d_norm * v_rel_norm), -1.0, 1.0)
        #     phi = np.arccos(cos_phi)

        #     # Time to Collision (TTC)
        #     TTC = -np.dot(d, v_rel) / (v_rel_norm ** 2)

        #     # 判斷是否為潛在碰撞（小角度 + TTC 為正）
        #     is_collision = phi < epsilon and TTC > 0

        #     # 預測未來碰撞點（可選）
        #     future_pos = p_j + v_j * TTC if is_collision else None #if TTC > 0 else None

        #     collisions.append({
        #         "id": group.group_id,
        #         "TTC": TTC,
        #         "phi": phi,
        #         "future_pos": future_pos,
        #         "is_collision": is_collision,
        #     })

        # rospy.logdebug(f"[SocialRuleSelector] Collision details: {collisions}")

        # return any(c['is_collision'] for c in collisions), collisions

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
        accum_dist = 0.0  # 累計距離

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
            accum_dist += dist
            timed_path_odom.append((accum_time, nx_odom, ny_odom))
            if accum_dist > 5.0:
                break
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
                        # rospy.loginfo("Collision detected!")
                        # rospy.loginfo(f"Robot Position: ({ox}, {oy}), Velocity: ({vx}, {vy})")
                        # rospy.loginfo(f"Pedestrian Group Position: ({traj.predicted_center.pose.position.x}, {traj.predicted_center.pose.position.y}), Velocity: ({pvx}, {pvy})")
                        # rospy.loginfo(f"Predicted Pedestrian Position: ({px}, {py}), Time: {t_pred.to_sec()}")
                        # rospy.loginfo(f"Global Path Point: ({x_odom}, {y_odom})")
                        return True, accum_time_odom

        return False, None

    def sampleTrajectories(self, collisions, sample_resolution = 0.5):
        rospy.logdebug(f"[SocialRuleSelector] Sampling trajectories")
        trajs = {
            "decelerate": [],
            "accelerate": [],
            "turn_right": [],
            "turn_left": []
        }

        ### Sample endpoint in Frenet coordinates

        # Calculate cumulative arc length
        path_pts = [(p.pose.position.x, p.pose.position.y) for p in self.global_path.poses]
        if (len(path_pts) < 2):
            rospy.logwarn(f"[SocialRuleSelector] Global path too short for trajectory sampling")
            return trajs
        theta0 = np.arctan2(path_pts[1][1] - path_pts[0][1], path_pts[1][0] - path_pts[0][0])
        cumulative_s = np.zeros(len(path_pts))
        for i in range(1, len(path_pts)):
            cumulative_s[i] = cumulative_s[i-1] + np.linalg.norm(np.array(path_pts[i]) - np.array(path_pts[i-1]))

        types = {
            "decelerate": {"s":(0.5, 2.0), "d":(-0.5, 0.5)}, #decelerate, num = 4 * 3 = 12
            "accelerate": {"s":(2.0, 4.0), "d":(-0.5, 0.5)}, #accelerate, num = 5 * 3 = 15
            "turn_right": {"s":(1.0, 3.0), "d":(-2.0, -0.5)}, #turn_right, num = 5 * 4 = 20
            "turn_left": {"s":(1.0, 3.0), "d":(0.5, 2.0)}, #turn_left, num = 5 * 4 = 20
        }

        for t_name, t in types.items():
            # Generate endpoints in Frenet coordinates
            endpoints_frenet = []
            s_range = t["s"]
            d_range = t["d"]
            for s in np.arange(s_range[0], s_range[1], sample_resolution):
                for d in np.arange(d_range[0], d_range[1], sample_resolution):
                    endpoints_frenet.append((s, d))

            # Transform endpoints from Frenet coordinate to Cartesian coordinate
            endpoints_cartesian = []
            for s, d in endpoints_frenet:
                # Find the corresponding point on the global path
                idx = np.searchsorted(cumulative_s, s) - 1
                idx = np.clip(idx, 0, len(path_pts) - 2)

                # Interpolate to get the global point
                ratio = (s - cumulative_s[idx]) / (cumulative_s[idx + 1] - cumulative_s[idx] + 1e-6)
                x = path_pts[idx][0] + ratio * (path_pts[idx + 1][0] - path_pts[idx][0])
                y = path_pts[idx][1] + ratio * (path_pts[idx + 1][1] - path_pts[idx][1])

                # Compute the tangent vector
                dx = path_pts[idx + 1][0] - path_pts[idx][0]
                dy = path_pts[idx + 1][1] - path_pts[idx][1]
                theta = np.arctan2(dy, dx)

                # Compute the normal vector
                normal = np.array([-np.sin(theta), np.cos(theta)])
                
                # Compute the endpoint in Cartesian coordinates
                endpoint = np.array([x, y]) + d * normal
                endpoints_cartesian.append(((endpoint, theta), s))

            # Generate trajectory
            for (endpoint, theta), s in endpoints_cartesian:
                traj = self.generate_trajectory((path_pts[0][0], path_pts[0][1], theta0), (endpoint[0], endpoint[1], theta))
                trajs[t_name].append((traj, s))

        return trajs

    def generate_trajectory(self, x_I, x_F, dt=0.1, steps=50,
                            K_rho=1.0, K_alpha=2.5, K_phi=-0.5):
        """
        x_I, x_F: (x, y, theta) in radians
        returns: list of (x, y, theta)
        """
        traj = [x_I]
        x, y, theta = x_I
        x_goal, y_goal, theta_goal = x_F

        for _ in range(steps):
            # 計算誤差（轉到目標座標系）
            dx = x_goal - x
            dy = y_goal - y
            rho = np.hypot(dx, dy)  # 與目標距離

            goal_angle = np.arctan2(dy, dx)
            alpha = np.arctan2(np.sin(goal_angle - theta),
                            np.cos(goal_angle - theta))
            # phi = np.arctan2(np.sin(goal_angle - theta_goal),
            #                 np.cos(goal_angle - theta_goal))
            phi = np.arctan2(np.sin(theta_goal - theta), np.cos(theta_goal - theta))

            # 控制律
            v = K_rho * rho
            omega = K_alpha * alpha + K_phi * phi

            # 限制最大速度（可選）
            v = np.clip(v, -1.0, 1.0)
            omega = np.clip(omega, -1.0, 1.0)

            # 模擬一步
            x += v * np.cos(theta) * dt
            y += v * np.sin(theta) * dt
            theta += omega * dt
            theta = np.arctan2(np.sin(theta), np.cos(theta))  # 正規化

            traj.append((x, y, theta))

        return traj

    def removeInvalidTrajs(self, trajs):
        rospy.logdebug(f"[SocialRuleSelector] Removing invalid trajectories")

        # Remove invalid trajectories (e.g., those that collide with obstacles)
        valid_trajs = {}
        self.predicted_groups = []
        steps = 50
        dt = 0.1
        robot_radius = 0.3

        try:
            trans = self.tf_buffer.lookup_transform(self.map_frame, 'odom', rospy.Time(0), rospy.Duration(1.0))
        except (tf2_ros.TransformException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"Transform from odom to map failed: {e}")
            return

        t = trans.transform.translation
        trans_vec = np.array([t.x, t.y, t.z])
        q = trans.transform.rotation
        rot_mat = tft.quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]

        for group in self.groups_state.groups:
            x0 = group.centerOfGravity.pose.position.x
            y0 = group.centerOfGravity.pose.position.y
            vx = group.group_velocity.linear.x
            vy = group.group_velocity.linear.y

            # 在 odom frame 預測
            traj_odom = np.zeros((steps, 3))
            traj_odom[:, 0] = x0 + vx * np.arange(steps) * dt
            traj_odom[:, 1] = y0 + vy * np.arange(steps) * dt
            traj_odom[:, 2] = 0.0  # z=0

            traj_map = (traj_odom @ rot_mat.T) + trans_vec

            self.predicted_groups.append((traj_map[:, :2], group.group_radius))

        for t_name, traj_list in trajs.items():
            valid_trajs[t_name] = []
            for traj, s in traj_list:
                collision = False
                for i, (x, y, theta) in enumerate(traj):
                    # ---- 靜態障礙檢查 ----
                    gx = int((x - self.map_origin[0]) / self.map_info.resolution)
                    gy = int((y - self.map_origin[1]) / self.map_info.resolution)
                    if gx < 0 or gx >= self.map_info.width or gy < 0 or gy >= self.map_info.height:
                        collision = True
                        break
                    if self.dist_field[gy, gx] < robot_radius:
                        collision = True
                        break

                    # ---- 動態群組檢查 ----
                    for (pred_traj, group_r) in self.predicted_groups:
                        if i < len(pred_traj):
                            gx, gy = pred_traj[i]
                            if np.hypot(x - gx, y - gy) < (robot_radius + group_r):
                                collision = True
                                break
                    if collision: 
                        break

                if not collision:
                    valid_trajs[t_name].append((traj, s))

        return valid_trajs
    
    def computeCosts(self, trajs):
        rospy.logdebug(f"[SocialRuleSelector] Computing costs for trajectories")

        costs = {}
        for t_name, traj_list in trajs.items():
            total_cost = 0.0
            for traj, s in traj_list:
                cost = 0.0
                # 1. Progress
                c_progress = s
                # 2. Similarity #TODO:maybe do not need this
                c_similarity = 0.0
                # 3. Proximity : Calculate the trajectory maximum distance to human at every moment
                for i, (x, y, theta) in enumerate(traj):
                    min_dist = float('inf')
                    for (pred_traj, group_r) in self.predicted_groups:
                        if i < len(pred_traj):
                            gx, gy = pred_traj[i]
                            dist = np.hypot(x - gx, y - gy)
                            if dist < min_dist:
                                min_dist = dist
                    if i == 0:
                        c_proximity = min_dist
                    else:
                        if min_dist < c_proximity:
                            c_proximity = min_dist
                c_proximity = np.exp(-c_proximity)

                cost = (self.weight_progress * c_progress +
                         self.weight_similarity * c_similarity +
                         self.weight_proximity * c_proximity)
                total_cost += cost
            costs[t_name] = total_cost / len(traj_list) if len(traj_list) > 0 else float('inf')
        return costs


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
        marker_array = MarkerArray()
        marker_id = 0
        color_map = {
            "decelerate": (1.0, 0.0, 0.0), # Red
            "accelerate": (0.0, 1.0, 0.0), # Green
            "turn_right": (1.0, 1.0, 0.0),   # Yellow
            "turn_left": (0.0, 0.0, 1.0),    # Blue
        }

        for t_name, traj_list in trajs.items():
            r, g, b = color_map.get(t_name, (1.0, 1.0, 1.0))
            for traj, s in traj_list:
                marker = Marker()
                marker.header.frame_id = self.map_frame
                marker.header.stamp = rospy.Time.now()
                marker.ns = "sampled_trajectory"
                marker.id = marker_id
                marker.type = Marker.LINE_STRIP
                marker.action = Marker.ADD
                marker.pose.orientation.w = 1.0
                marker.scale.x = 0.05  # 線寬
                marker.color.a = 1.0
                marker.color.r = r
                marker.color.g = g
                marker.color.b = b
                marker.lifetime = rospy.Duration(5)  # 持續時間

                for point in traj:
                    p = Point()
                    p.x = point[0]
                    p.y = point[1]
                    p.z = 0.0
                    marker.points.append(p)

                marker_array.markers.append(marker)
                marker_id += 1

        self.pub_sample_trajs.publish(marker_array)

if __name__=='__main__':
    rospy.init_node('social_rule_selector')
    node = SocialRuleSelector()
    node.run()
