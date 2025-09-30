#ifndef SOCIAL_RULE_LAYER_H
#define SOCIAL_RULE_LAYER_H

#include <ros/ros.h>
#include <costmap_2d/layer.h>
#include <costmap_2d/layered_costmap.h>
#include <costmap_2d/costmap_layer.h>
#include <geometry_msgs/PointStamped.h>
#include <nav_msgs/Odometry.h>
#include <tf2/utils.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <social_rules_selector/TrackedGroups.h>
#include <social_rules_selector/SocialRule.h>
#include <mutex>
#include <vector>
#include <Eigen/Dense>
#include <cmath>

using namespace std;

// #define M_PI 3.14159265359

namespace social_rule_layer
{

class SocialRuleLayer : public costmap_2d::CostmapLayer
{
public:
  SocialRuleLayer();
  virtual ~SocialRuleLayer();

  virtual void onInitialize();
  virtual void updateBounds(double robot_x, double robot_y, double robot_yaw,
                            double *min_x, double *min_y, double *max_x, double *max_y);
  virtual void updateCosts(costmap_2d::Costmap2D &master_grid,
                           int min_i, int min_j, int max_i, int max_j);
  virtual void activate();
  virtual void deactivate();
  virtual void reset();

private:
  struct TransformGroup
  {
    geometry_msgs::Pose centerOfGravity;
    geometry_msgs::Twist group_velocity;
    double group_radius;
    std::vector<uint64_t> track_ids;
  };
  
  void trackedGroupsCallback(const social_rules_selector::TrackedGroups::ConstPtr& msg);
  void socialRuleCallback(const social_rules_selector::SocialRule::ConstPtr& msg);
  void odomCallback(const nav_msgs::Odometry::ConstPtr& msg);
  double normalize(double angle);
  double ComputeSigma(double variance);
  double Asymmetrical_Gaussian(double x, double y, double x0, double y0, double vx, double vy, double group_radius);
  double ApplySocialRuleCost(double x, double y, double x0, double y0, double vx, double vy, double group_radius);

  ros::Subscriber tracked_groups_sub_;
  ros::Subscriber social_rule_sub_;
  ros::Subscriber odom_sub_;
  ros::NodeHandle nh_;

  // Data storage
  social_rules_selector::TrackedGroups current_groups_;
  social_rules_selector::SocialRule current_rule_;
  nav_msgs::Odometry current_odom_;

  std::mutex data_mutex_;  // Protect access to current_groups_ and current_rule_

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;


  bool first_time_;
  double last_min_x_, last_min_y_, last_max_x_, last_max_y_;
  vector<TransformGroup> transformed_groups_;
};

} // namespace social_rules_selector

#endif // SOCIAL_RULE_LAYER_H
