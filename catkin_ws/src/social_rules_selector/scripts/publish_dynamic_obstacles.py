#!/usr/bin/env python
import rospy
from costmap_converter.msg import ObstacleArrayMsg, ObstacleMsg
from geometry_msgs.msg import Point32
from ultralytics_ros.msg import Trk3DArray  # 你的自定義訊息

def trk3d_callback(msg):
    obs_array = ObstacleArrayMsg()
    obs_array.header = msg.header  # 沿用追蹤訊息的時間戳和 frame_id

    for trk in msg.trks_list:
        obs = ObstacleMsg()
        obs.id = trk.tracked_id
        obs.polygon.points.append(Point32(trk.x, trk.y, 0.0))  # 中心點
        obs.radius = trk.radius if trk.radius > 0.0 else 0.3   # 半徑，避免 0

        # 設定速度
        obs.velocities.twist.linear.x = trk.vx
        obs.velocities.twist.linear.y = trk.vy

        obs_array.obstacles.append(obs)

    pub.publish(obs_array)

if __name__ == "__main__":
    rospy.init_node("publish_dynamic_obstacles_node", anonymous=True)
    pub = rospy.Publisher("/move_base/TebLocalPlannerROS/obstacles",
                          ObstacleArrayMsg, queue_size=10)

    rospy.Subscriber("/trk3d_result", Trk3DArray, trk3d_callback, queue_size=10)

    rospy.loginfo("Trk3D -> TebLocalPlannerROS/obstacles bridge running...")
    rospy.spin()
