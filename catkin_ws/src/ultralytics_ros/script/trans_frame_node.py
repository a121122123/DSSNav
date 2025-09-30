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
from ultralytics_ros.msg import Det3DArray, Trk3DArray, Det3D, Trk3D
from kalman_tracker import MultiObjectTracker

class Yolo3DTransformNode:
    def __init__(self):
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(2.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.pub_pose = rospy.Publisher("transformed_bbox_center", Point, queue_size=10)
        self.pub_trk = rospy.Publisher("trk3d_result", Trk3DArray, queue_size=10)
        self.pub_marker = rospy.Publisher("trk3d_visual", MarkerArray, queue_size=10)
        rospy.Subscriber("yolo_3d_result", Det3DArray, self.detection_callback)
        self.tracker = MultiObjectTracker()
        self.last_time = None
        self.duration = 1.0  # Default duration for the first run

    def detection_callback(self, msg):
        trk3d_array = Trk3DArray()
        trk3d_array.header = Header()
        trk3d_array.header.frame_id = "odom"
        trk3d_array.header.stamp = msg.header.stamp
        marker_array = MarkerArray()

        for detection in msg.detections:
            # Create a PointStamped message for the bounding box center
            bbox_center = detection.center.position
            bbox_center_stamped = PointStamped()
            bbox_center_stamped.header = msg.header  # Set time stamp and frame id(it should be the cloud msg frame id)
            bbox_center_stamped.point = Point(bbox_center.x, bbox_center.y, bbox_center.z)
            
            try:
                bbox_center_odom_stamped = self.tf_buffer.transform(bbox_center_stamped, "odom", rospy.Duration(1.0))

                if detection.tracked_id != -1:
                    self.tracker.update_object(detection.tracked_id, detection.class_name, detection.score, 
                                               bbox_center_odom_stamped.point.x, bbox_center_odom_stamped.point.y, 
                                               max(detection.size.x, detection.size.y) / 2, 
                                               self.duration)
                # TODO: object_id -1 is not tracked, but we still need to add it to the tracker
                # else:
                #     self.tracker.add_object(detection.tracked_id, detection.class_name, detection.score, bbox_center_odom_stamped.point.x, bbox_center_odom_stamped.point.y, max(detection.size.x, detection.size.y) / 2)
                    
                self.pub_pose.publish(bbox_center_odom_stamped.point)
            except tf2_ros.LookupException as ex:
                rospy.logwarn("TF LookupException: %s", ex)
            except tf2_ros.ExtrapolationException as ex:
                rospy.logwarn("TF ExtrapolationException: %s", ex)       
        
        # self.tracker.predict_object(self.duration)  # Predict the next state for all tracked objects
        self.tracker.remove_old_objects()

        # Initialize the last_time variable
        if self.last_time is None:
            self.last_time = rospy.Time.now()
            return
        current_time = rospy.Time.now()
        self.duration = (current_time - self.last_time).to_sec()
        # rospy.loginfo("Time since last update: %.3f seconds", self.duration)
        if self.duration < 0.04: # scan頻率
            self.duration = 0.04  # Ensure a minimum duration to avoid division by zero
        self.last_time = current_time

        # Create a Marker message for visualization
        for object_id, tracker in self.tracker.trackers.items():
            x, y, vx, vy = tracker.get_state()

            trk3d = Trk3D()
            trk3d.tracked_id = object_id
            trk3d.class_name = tracker.class_name
            trk3d.x = x
            trk3d.y = y
            if math.hypot(vx / self.duration, vy / self.duration) > 0.4 and tracker.validity:
                trk3d.vx = vx / self.duration
                trk3d.vy = vy / self.duration
                trk3d.yaw = np.arctan2(vy, vx)
            else:
                trk3d.vx = 0
                trk3d.vy = 0
                trk3d.yaw = 0
            trk3d.radius = tracker.r
            trk3d.confidence = tracker.confidence
            trk3d_array.trks_list.append(trk3d)

            # Visualize the bounding box as a cylinder
            marker = Marker()
            marker.header.frame_id = "odom"
            marker.header.stamp = msg.header.stamp
            marker.ns = "people"
            marker.id = object_id
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = 0.8
            quat = tf_trans.quaternion_from_euler(0, 0, trk3d.yaw)
            marker.pose.orientation = Quaternion(*quat)
            marker.scale.x = tracker.r * 2.0
            marker.scale.y = tracker.r * 2.0
            marker.scale.z = 1.6
            marker.color.a = 0.5
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.lifetime = rospy.Duration(1.0)
            marker_array.markers.append(marker)

        # Publish the tracked result
        self.pub_trk.publish(trk3d_array)
        # Publish the marker
        self.pub_marker.publish(marker_array)
                

if __name__ == '__main__':
    rospy.init_node('yolo_3d_transform_node')
    node = Yolo3DTransformNode()
    rospy.spin()
