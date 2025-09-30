#!/usr/bin/env python3
import rospy
import tf2_ros
import numpy as np
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import TransformStamped
from ultralytics_ros.msg import Trk3DArray  # 改成你自己的訊息類型

class HumanFilter(object):
    def __init__(self):
        rospy.init_node('human_filter')
        self.rate = rospy.Rate(50.0)
        self.humans = []
        self.laser_transform = TransformStamped()

        rospy.Subscriber('/scan', LaserScan, self.laserCB)
        rospy.Subscriber('/trk3d_result', Trk3DArray, self.humansCB)
        self.laser_pub = rospy.Publisher('/scan_filtered', LaserScan, queue_size=10)

        self.tf = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf)

        rospy.spin()

    def laserCB(self, scan):
        filtered_scan = scan
        filtered_scan.ranges = list(scan.ranges)
        filtered_scan.header.stamp = rospy.Time.now()
        # rospy.loginfo(f"Scan angle range: {np.degrees(scan.angle_min):.1f}° ~ {np.degrees(scan.angle_max):.1f}°")

        try:
            self.laser_transform = self.tf.lookup_transform('odom', 'base_scan', rospy.Time(), rospy.Duration(0.5))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logwarn("Failed to get transform from odom to base_scan, skipping filtering.")
            self.rate.sleep()
            return

        if self.laser_transform.header.frame_id != '':
            laser_pose = self.laser_transform.transform.translation
            rot = self.laser_transform.transform.rotation
            _, _, yaw = euler_from_quaternion([rot.x, rot.y, rot.z, rot.w])
            base_laser_dir = [np.cos(yaw), np.sin(yaw)]

            for human in self.humans:
                rh_vec = [human[0] - laser_pose.x, human[1] - laser_pose.y]
                if np.linalg.norm(rh_vec) < 0.01:
                    rospy.logwarn("Ignoring human too close to laser: %s", human)
                    continue
                # sign = base_laser_dir[0]*-rh_vec[1] + base_laser_dir[1]*rh_vec[0]
                # sign = sign / abs(sign)
                # t_angle = scan.angle_max - scan.angle_min
                # mid_angle = t_angle / 2 - sign * np.arccos((base_laser_dir[0]*rh_vec[0]+base_laser_dir[1]*rh_vec[1]) / np.linalg.norm(rh_vec))
                # rospy.loginfo(f"Human at {human} with mid angle {np.degrees(mid_angle):.1f}°")
                angle = np.arctan2(rh_vec[1], rh_vec[0]) - yaw
                # normalize angle to [-pi, pi]
                # angle = np.arctan2(np.sin(angle), np.cos(angle))
                if angle < 0:
                    angle += 2 * np.pi
                rospy.logdebug(f"Human at {human} with angle {np.degrees(angle):.1f}°")
                mid_idx = int((angle - scan.angle_min) / scan.angle_increment)
                # mid_idx = int(mid_angle / scan.angle_increment)
                if mid_idx >= len(scan.ranges):
                    rospy.logdebug("Mid index %d out of range for scan with %d ranges, skipping filtering.", mid_idx, len(scan.ranges))
                    continue

                r = 0.25  # 遮蔽半徑，可調整
                d = np.linalg.norm(rh_vec)
                mr = scan.ranges[mid_idx]
                if mr <= (d - r):
                    rospy.logdebug(f"Human at {human} is too far, skipping filtering.")
                    rospy.logdebug(f"Distance to human: {d}, range at mid index: {mr}")
                    continue

                if r <= d:
                    beta = np.arcsin(r / d)
                else:
                    beta = np.pi / 2

                # min_idx = int(np.floor((mid_angle - beta) / scan.angle_increment))
                # max_idx = int(np.ceil((mid_angle + beta) / scan.angle_increment))
                min_idx = int(np.floor((angle - beta) / scan.angle_increment))
                max_idx = int(np.ceil((angle + beta) / scan.angle_increment))

                for i in range(min_idx, max_idx):
                    if 0 <= i < len(scan.ranges):
                        filtered_scan.ranges[i] = float('NaN')
                rospy.logdebug(f"Filtered idx from {min_idx} to {max_idx} for human at {human}")
        else:
            rospy.logdebug("Laser transform frame_id is empty, skipping filtering.")

        self.laser_pub.publish(filtered_scan)

    def humansCB(self, msg):
        self.humans = []
        for human in msg.trks_list:
            self.humans.append((human.x, human.y))
        rospy.logdebug(f"Received {len(self.humans)} humans")

if __name__ == '__main__':
    hfilter = HumanFilter()