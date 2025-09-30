#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Point32, Twist
from costmap_converter.msg import ObstacleArrayMsg, ObstacleMsg

def publish_obstacle():
    pub = rospy.Publisher("/move_base/TebLocalPlannerROS/obstacles",
                          ObstacleArrayMsg, queue_size=10)
    rospy.init_node("test_obstacle_publisher", anonymous=True)
    rate = rospy.Rate(10)  # 10 Hz

    while not rospy.is_shutdown():
        msg = ObstacleArrayMsg()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "odom"   # 或者 "map"，要和 TEB 的 frame 對齊

        obs = ObstacleMsg()
        obs.id = 1
        obs.polygon.points.append(Point32(3.5, 2.0, 0.0))  # 障礙物中心點
        obs.radius = 0.3                                   # 半徑

        # 設定速度 (朝 -x 移動)
        obs.velocities.twist.linear.x = -0.5
        obs.velocities.twist.linear.y = 0.0

        msg.obstacles.append(obs)

        pub.publish(msg)
        rate.sleep()

if __name__ == "__main__":
    try:
        publish_obstacle()
    except rospy.ROSInterruptException:
        pass
