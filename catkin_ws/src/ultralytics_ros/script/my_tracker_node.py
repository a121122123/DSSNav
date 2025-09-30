#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ultralytics_ros
# Copyright (C) 2023-2024  Alpaca-zip
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import cv_bridge
import numpy as np
import roslib.packages
import rospy
from sensor_msgs.msg import Image, LaserScan, PointCloud2
from ultralytics import YOLO
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from ultralytics_ros.msg import YoloResult, MyYoloResult, BBox2D, BBox2DArray
import laser_geometry.laser_geometry as lg

import yaml
from ultralytics.trackers.byte_tracker import BYTETracker

class TrackerArgs:
    def __init__(self, config):
        for key, value in config.items():
            setattr(self, key, value)


class TrackerNode:
    def __init__(self):
        # YOLO model parameters
        yolo_model = rospy.get_param("~yolo_model", "yolov8n.pt")
        self.input_topic = rospy.get_param("~input_topic", "image_raw")
        self.result_topic = rospy.get_param("~result_topic", "yolo_result")
        self.result_image_topic = rospy.get_param("~result_image_topic", "yolo_image")
        self.pointcloud_topic = rospy.get_param("~pointcloud_topic", "converted_pc")
        self.conf_thres = rospy.get_param("~conf_thres", 0.25)
        self.iou_thres = rospy.get_param("~iou_thres", 0.45)
        self.max_det = rospy.get_param("~max_det", 300)
        self.classes = rospy.get_param("~classes", None)
        self.tracker = rospy.get_param("~tracker", "bytetrack.yaml")
        self.device = rospy.get_param("~device", "cuda:0")
        self.result_conf = rospy.get_param("~result_conf", True)
        self.result_line_width = rospy.get_param("~result_line_width", None)
        self.result_font_size = rospy.get_param("~result_font_size", None)
        self.result_font = rospy.get_param("~result_font", "Arial.ttf")
        self.result_labels = rospy.get_param("~result_labels", True)
        self.result_boxes = rospy.get_param("~result_boxes", True)
        self.path = roslib.packages.get_pkg_dir("ultralytics_ros")
        self.model = YOLO(f"{self.path}/models/{yolo_model}")
        self.model.fuse()

        # Tracker setup
        with open(f"{self.path}/cfg/{self.tracker}", "r") as file:
            tracker_config = yaml.safe_load(file)

        args = TrackerArgs(tracker_config)

        self.mytracker = BYTETracker(args)

        # Laser to PointCloud2 setup
        self.laser_projector = lg.LaserProjection()

        # Subscriptions and Publishers
        self.image_sub = rospy.Subscriber(
            self.input_topic,
            Image,
            self.image_callback,
            queue_size=1,
            buff_size=2**24,
        )
        self.laser_sub = rospy.Subscriber(
            "/scan",
            LaserScan,
            self.laser_callback,
            queue_size=1,
        )
        # self.results_pub = rospy.Publisher(self.result_topic, YoloResult, queue_size=1)
        self.results_pub = rospy.Publisher(self.result_topic, MyYoloResult, queue_size=1)
        self.result_image_pub = rospy.Publisher(
            self.result_image_topic, Image, queue_size=1
        )
        self.pointcloud_pub = rospy.Publisher(self.pointcloud_topic, PointCloud2, queue_size=1)
        self.bridge = cv_bridge.CvBridge()
        self.use_segmentation = yolo_model.endswith("-seg.pt")

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        mytracker = self.mytracker

        results = self.model.track(
            source=cv_image,
            conf=self.conf_thres,
            iou=self.iou_thres,
            max_det=self.max_det,
            classes=self.classes,
            tracker=f"{self.path}/cfg/{self.tracker}", 
            device=self.device,
            verbose=False,
            retina_masks=True,
            persist=True,
            agnostic_nms=True,
            # augment=True,
            # visualize=True,
        )

        if results is not None:
            '''
            yolo_result_msg = YoloResult()
            yolo_result_image_msg = Image()
            yolo_result_msg.header = msg.header
            yolo_result_image_msg.header = msg.header
            yolo_result_msg.detections = self.create_detections_array(results)
            yolo_result_image_msg = self.create_result_image(results)
            if self.use_segmentation:
                yolo_result_msg.masks = self.create_segmentation_masks(results)
            self.results_pub.publish(yolo_result_msg)
            self.result_image_pub.publish(yolo_result_image_msg)
            '''

            my_yolo_result_msg = MyYoloResult()
            my_yolo_result_msg.header = msg.header
            my_yolo_result_msg.detections = self.my_create_detections_array(results)
            self.results_pub.publish(my_yolo_result_msg)

            yolo_result_image_msg = Image()
            yolo_result_image_msg.header = msg.header
            yolo_result_image_msg = self.create_result_image(results)
            self.result_image_pub.publish(yolo_result_image_msg)

    def laser_callback(self, msg):
        # Convert LaserScan to PointCloud2
        pc2_msg = self.laser_projector.projectLaser(msg)

        # Publish PointCloud2
        self.pointcloud_pub.publish(pc2_msg)

    '''
    def create_detections_array(self, results):
        detections_msg = Detection2DArray()
        bounding_box = results[0].boxes.xywh
        classes = results[0].boxes.cls
        confidence_score = results[0].boxes.conf
        for bbox, cls, conf in zip(bounding_box, classes, confidence_score):
            detection = Detection2D()
            detection.bbox.center.x = float(bbox[0])
            detection.bbox.center.y = float(bbox[1])
            detection.bbox.size_x = float(bbox[2])
            detection.bbox.size_y = float(bbox[3])
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.id = int(cls)
            hypothesis.score = float(conf)
            detection.results.append(hypothesis)
            detections_msg.detections.append(detection)
        return detections_msg
    '''
    
    def my_create_detections_array(self, results):
        """
        :param results: YOLO 偵測結果
        :return: BBox2DArray 訊息
        """
        detections_msg = BBox2DArray()
        bounding_box = results[0].boxes.xywh
        classes = results[0].boxes.cls
        confidence_score = results[0].boxes.conf
        tracked = results[0].boxes.is_track
        if isinstance(tracked, bool):  
            tracked = [tracked] * len(bounding_box)  # 轉換成與 bounding_box 長度相符的列表

        # 檢查 id 是否存在，若不存在則用 -1 填充
        id = getattr(results[0].boxes, "id", None)  # 安全獲取 id，避免 AttributeError
        if id is None:
            id = [-1] * len(bounding_box)  # 給每個偵測物一個預設 id

        for bbox, cls, conf, track, id in zip(bounding_box, classes, confidence_score, tracked, id):
            detection = BBox2D()
            detection.center.x = float(bbox[0])
            detection.center.y = float(bbox[1])
            detection.size_x = float(bbox[2])
            detection.size_y = float(bbox[3])
            detection.class_name = results[0].names[int(cls)]
            detection.score = float(conf)
            detection.tracked = bool(track)
            detection.id = int(id)
            detections_msg.bboxes.append(detection)
        return detections_msg

    def create_result_image(self, results):
        plotted_image = results[0].plot(
            conf=self.result_conf,
            line_width=self.result_line_width,
            font_size=self.result_font_size,
            font=self.result_font,
            labels=self.result_labels,
            boxes=self.result_boxes,
        )
        result_image_msg = self.bridge.cv2_to_imgmsg(plotted_image, encoding="bgr8")
        return result_image_msg

    '''
    def create_segmentation_masks(self, results):
        masks_msg = []
        for result in results:
            if hasattr(result, "masks") and result.masks is not None:
                for mask_tensor in result.masks:
                    mask_numpy = (
                        np.squeeze(mask_tensor.data.to("cpu").detach().numpy()).astype(
                            np.uint8
                        )
                        * 255
                    )
                    mask_image_msg = self.bridge.cv2_to_imgmsg(
                        mask_numpy, encoding="mono8"
                    )
                    masks_msg.append(mask_image_msg)
        return masks_msg
    '''

if __name__ == "__main__":
    rospy.init_node("tracker_node")
    node = TrackerNode()
    rospy.spin()
