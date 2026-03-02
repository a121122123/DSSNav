#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2, socket, json
import cv_bridge
import numpy as np
import roslib.packages
import rospy
from sensor_msgs.msg import Image, LaserScan, PointCloud2
from ultralytics import YOLO
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from ultralytics_ros.msg import YoloResult, MyYoloResult, BBox2D, BBox2DArray
import laser_geometry.laser_geometry as lg


sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(("172.17.0.1", 5001))


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


    def laser_callback(self, msg):
        # Convert LaserScan to PointCloud2
        pc2_msg = self.laser_projector.projectLaser(msg)

        # Publish PointCloud2
        self.pointcloud_pub.publish(pc2_msg)

    
    def image_callback(self,msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        #ok, encoded_img = cv2.imencode('.jpg', cv_image)
        #if not ok:
         #   rospy.logerr("[TrackerNode] Failed to encode image")
        #    return
        
        # Send image to host for YOLO inference
        try:
            sock.send(len(cv_image.tobytes()).to_bytes(4, 'big'))
            sock.sendall(cv_image.tobytes())
        except Exception as e:
            rospy.logerr(f"[TrackerNode] Failed to send image to host: {e}")
            return

        # Receive YOLO inference results from host
        try:
            length_bytes = sock.recv(4)
            length = int.from_bytes(length_bytes, 'big')
            json_bytes = sock.recv(length, socket.MSG_WAITALL)
            json_msg = json_bytes.decode()
            data = json.loads(json_msg)
        except Exception as e:
            rospy.logerr(f"[TrackerNode] Failed to receive YOLO results: {e}")
            return
        
        # Convert to MyYoloResult message
        my_yolo_result_msg = MyYoloResult()
        my_yolo_result_msg.header = msg.header

        detections_msg = BBox2DArray()
        for det in data["detections"]:
            cx, cy, w, h = det["cx"], det["cy"], det["w"], det["h"]
            cls, conf, track_id = det["cls"], det["conf"], det["id"]

            # Draw bounding box on image (optional)
            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)
            # bbox & label
            cv2.rectangle(cv_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            label = f"ID:{track_id} {cls} {conf:.2f}"
            # measure text size
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            h = h + 3 # small padding
            # check if label fits above bbox (outside = above)
            outside = y1 >= h
            # ensure label does not go beyond right border
            if x1 + w > cv_image.shape[1]:
                x1 = cv_image.shape[1] - w
            # define background rectangle for label
            # p1 = top-left, p2 = bottom-right
            p1 = (x1, y1)
            p2 = (x1 + w, y1 - h) if outside else (x1 + w, y1 + h)
            # p1 = (x1, y1 - h if outside else y1)
            # p2 = (x1 + w, y1) if outside else (x1 + w, y1 + h)
            # draw filled label background
            cv2.rectangle(cv_image, p1, p2, (0, 255, 0), -1, cv2.LINE_AA)

            cv2.putText(cv_image, label, (p1[0], p1[1] - 2 if outside else p1[1] + h - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)


            # Create BBox2D message
            detection = BBox2D()
            detection.center.x = cx
            detection.center.y = cy
            detection.size_x = w
            detection.size_y = h
            detection.class_name = cls
            detection.score = conf
            detection.id = track_id
            detections_msg.bboxes.append(detection)

        my_yolo_result_msg.detections = detections_msg
        self.results_pub.publish(my_yolo_result_msg)
        # Publish result image with bounding boxes
        result_image_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
        result_image_msg.header = msg.header
        self.result_image_pub.publish(result_image_msg)

if __name__ == "__main__":
    rospy.init_node("tracker_node")
    node = TrackerNode()
    rospy.spin()

