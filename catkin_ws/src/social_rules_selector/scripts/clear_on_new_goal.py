#!/usr/bin/env python3
import rospy
from std_srvs.srv import Empty
from geometry_msgs.msg import PoseStamped

class ClearOnNewGoal:
    def __init__(self):
        rospy.wait_for_service('/move_base/clear_costmaps')
        self.clear_srv = rospy.ServiceProxy('/move_base/clear_costmaps', Empty)
        self.last_goal = None
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback)

    def goal_callback(self, msg):
        if self.last_goal is None or self.has_goal_changed(msg):
            rospy.loginfo("New goal received, clearing costmaps...")
            try:
                self.clear_srv()
            except rospy.ServiceException as e:
                rospy.logwarn("Failed to clear costmaps: %s", e)
        self.last_goal = msg

    def has_goal_changed(self, new_goal):
        dx = self.last_goal.pose.position.x - new_goal.pose.position.x
        dy = self.last_goal.pose.position.y - new_goal.pose.position.y
        return (dx*dx + dy*dy) > 0.1  # 0.1 m 以上變化才算新目標

if __name__ == '__main__':
    rospy.init_node('clear_on_new_goal')
    ClearOnNewGoal()
    rospy.spin()
