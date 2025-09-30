#include "social_rule_layer/social_rule_layer.h"
#include <pluginlib/class_list_macros.h>

namespace social_rule_layer
{

SocialRuleLayer::SocialRuleLayer() : tf_listener_(tf_buffer_) {}

SocialRuleLayer::~SocialRuleLayer() {}

void SocialRuleLayer::onInitialize()
{
  ros::NodeHandle & nh = nh_;
  // Subscribe to tracked groups and social rule topics
  tracked_groups_sub_ = nh.subscribe("/tracked_groups", 1, &SocialRuleLayer::trackedGroupsCallback, this);
  social_rule_sub_ = nh.subscribe("/social_rule", 1, &SocialRuleLayer::socialRuleCallback, this);
  odom_sub_ = nh.subscribe("/odom", 1, &SocialRuleLayer::odomCallback, this);

  current_rule_.rule = "normal";

  matchSize();

  enabled_ = true;
  current_ = true;
  first_time_ = true;

  ROS_INFO("SocialRuleLayer initialized");
}

void SocialRuleLayer::updateBounds(double robot_x, double robot_y, double robot_yaw,
                                   double *min_x, double *min_y, double *max_x, double *max_y)
{
  std::lock_guard<std::mutex> lock(data_mutex_);
  // Get global frame ID, which is usually "map" or "odom". Can be checked in the costmap parameters yaml file.
  std::string global_frame = layered_costmap_->getGlobalFrameID();

  transformed_groups_.clear();

  // Convert group infos to global coordinates
  for (const auto& group : current_groups_.groups)
  {
    TransformGroup transformed_group;
    transformed_group.centerOfGravity = group.centerOfGravity.pose;
    transformed_group.group_velocity = group.group_velocity;
    transformed_group.group_radius = group.group_radius;
    transformed_group.track_ids = group.track_ids;

    // Transform the center of gravity to the global frame
    geometry_msgs::PointStamped group_point, transformed_point;
    try
    {
      group_point.header.frame_id = current_groups_.header.frame_id;
      group_point.header.stamp = current_groups_.header.stamp;
      group_point.point = group.centerOfGravity.pose.position;
      tf_buffer_.transform(group_point, transformed_point, global_frame, ros::Duration(0.5));
      transformed_group.centerOfGravity.position = transformed_point.point;

      group_point.point.x += group.group_velocity.linear.x;
      group_point.point.y += group.group_velocity.linear.y;
      tf_buffer_.transform(group_point, transformed_point, global_frame, ros::Duration(0.5));
      transformed_group.group_velocity.linear.x = transformed_point.point.x - transformed_group.centerOfGravity.position.x;
      transformed_group.group_velocity.linear.y = transformed_point.point.y - transformed_group.centerOfGravity.position.y;
      // ROS_INFO("Transformed group center: (%.2f, %.2f)", 
      //          transformed_group.centerOfGravity.position.x, transformed_group.centerOfGravity.position.y);
      // ROS_INFO("Transformed group velocity: (%.2f, %.2f)", 
      //          transformed_group.group_velocity.linear.x, transformed_group.group_velocity.linear.y);

      transformed_groups_.push_back(transformed_group);
    }
    catch (tf2::TransformException &ex)
    {
      ROS_WARN("Transform failed: %s", ex.what());
      continue; // Skip this group if transformation fails
    }
  }

  // Update the bounding box based on the transformed group center
  for (const auto& transformed_group : transformed_groups_)
  {
    // Assuming each group has a fixed radius, we can use it to expand the bounding box
    double radius = transformed_group.group_radius;
    double inflation = 3.5; // Offset to expand the bounding box
    *min_x = min(*min_x, transformed_group.centerOfGravity.position.x - radius - inflation);
    *min_y = min(*min_y, transformed_group.centerOfGravity.position.y - radius - inflation);
    *max_x = max(*max_x, transformed_group.centerOfGravity.position.x + radius + inflation);
    *max_y = max(*max_y, transformed_group.centerOfGravity.position.y + radius + inflation);
  }

  // If this is the first time, initialize the last bounds
  if (first_time_)
  {
    last_min_x_ = *min_x;
    last_min_y_ = *min_y;
    last_max_x_ = *max_x;
    last_max_y_ = *max_y;
    first_time_ = false;
  }
  else
  {
    last_min_x_ = *min_x;
    last_min_y_ = *min_y;
    last_max_x_ = *max_x;
    last_max_y_ = *max_y;
    // Expand the bounding box with the last bounds
    *min_x = std::min(last_min_x_, *min_x);
    *min_y = std::min(last_min_y_, *min_y);
    *max_x = std::max(last_max_x_, *max_x);
    *max_y = std::max(last_max_y_, *max_y);
  }
}

void SocialRuleLayer::updateCosts(costmap_2d::Costmap2D &master_grid,
                                  int min_i, int min_j, int max_i, int max_j)
{
  std::lock_guard<std::mutex> lock(data_mutex_);

  if (!current_ || transformed_groups_.empty())
  {
    // ROS_WARN("SocialRuleLayer is not current or has no transformed groups.");
    return;
  }

  costmap_2d::Costmap2D* costmap = layered_costmap_->getCostmap();
  double resolution = costmap->getResolution();
  double inflation = 4.0; // Offset to expand the cost area

  for (const auto& group : transformed_groups_)
  {
    // Group center and velocity (World coordinates)
    double group_x = group.centerOfGravity.position.x;
    double group_y = group.centerOfGravity.position.y;
    double group_velocity_x = group.group_velocity.linear.x;
    double group_velocity_y = group.group_velocity.linear.y;
    double inflation_radius = group.group_radius + inflation; // Offset to expand the cost area

    // Group left bottom corner (World coordinates)
    double area_min_x = group_x - inflation_radius;
    double area_min_y = group_y - inflation_radius;

    // Transform group left bottom corner to grid coordinates
    int costmap_min_x, costmap_min_y;
    costmap->worldToMapNoBounds(area_min_x, area_min_y, costmap_min_x, costmap_min_y);

    // Size of the grid area to be updated
    int costmap_size_width = max(1, static_cast<int>(ceil(inflation_radius * 2 / resolution)));
    int costmap_size_height = max(1, static_cast<int>(ceil(inflation_radius * 2 / resolution)));

    // Fix the bounds to ensure they are within the costmap limits
    int start_i = 0, start_j = 0, end_i = costmap_size_width, end_j = costmap_size_height;
    // X part
    if (costmap_min_x < 0) 
    {
      start_i = -costmap_min_x;
      // costmap_min_x = 0;
    }
    if (costmap_min_x + costmap_size_width > costmap->getSizeInCellsX()) 
    {
      end_i = max(0, static_cast<int>(costmap->getSizeInCellsX()) - costmap_min_x);
    }
    if (costmap_min_x + start_i < min_i) 
    {
      start_i = min_i - costmap_min_x;
    }
    if (costmap_min_x + end_i > max_i) 
    {
      end_i = max_i - costmap_min_x;
    }

    // Y part
    if (costmap_min_y < 0) 
    {
      start_j = -costmap_min_y;
      // costmap_min_y = 0;
    }
    if (costmap_min_y + costmap_size_height > costmap->getSizeInCellsY()) 
    {
      end_j = max(0, static_cast<int>(costmap->getSizeInCellsY()) - costmap_min_y);
    }
    if (costmap_min_y + start_j < min_j) 
    {
      start_j = min_j - costmap_min_y;
    }
    if (costmap_min_y + end_j > max_j) 
    {
      end_j = max_j - costmap_min_y;
    }

    double world_base_x = area_min_x + resolution / 2.0;
    double world_base_y = area_min_y + resolution / 2.0;

    // Update costs in the specified area
    for (int i = start_i; i < end_i; ++i)
    {
      for (int j = start_j; j < end_j; ++j)
      {
        double map_x = i + costmap_min_x;
        double map_y = j + costmap_min_y;
        double world_x = world_base_x + i * resolution;
        double world_y = world_base_y + j * resolution;

        unsigned char origin_cost = costmap->getCost(map_x, map_y);
        if (origin_cost == costmap_2d::NO_INFORMATION)
        {
          continue; // Skip if the cell is unknown
        }

        double cost_value = Asymmetrical_Gaussian(world_x, world_y, group_x, group_y, group_velocity_x, group_velocity_y, 
                                                  group.group_radius);

        unsigned char new_cost = static_cast<unsigned char>(cost_value);
        costmap->setCost(map_x, map_y, max(new_cost, origin_cost));
      }
    }
  }
}

void SocialRuleLayer::activate()
{
  // Called when layer is activated
}

void SocialRuleLayer::deactivate()
{
  // Called when layer is deactivated
}

void SocialRuleLayer::reset()
{
  // Reset internal state
}

void SocialRuleLayer::trackedGroupsCallback(const social_rules_selector::TrackedGroups::ConstPtr& msg)
{
  std::lock_guard<std::mutex> lock(data_mutex_);
  current_groups_ = *msg;
  // ROS_INFO("Received tracked groups: %ld", current_groups_.groups.size());
}

void SocialRuleLayer::socialRuleCallback(const social_rules_selector::SocialRule::ConstPtr& msg)
{
  std::lock_guard<std::mutex> lock(data_mutex_);
  current_rule_ = *msg;
  // ROS_INFO("Received social rule: %s", current_rule_.rule.c_str());
}

void SocialRuleLayer::odomCallback(const nav_msgs::Odometry::ConstPtr& msg)
{
  std::lock_guard<std::mutex> lock(data_mutex_);
  current_odom_ = *msg;
}

double SocialRuleLayer::normalize(double angle) {
  while (angle > M_PI) angle -= 2 * M_PI;
  while (angle <= -M_PI) angle += 2 * M_PI;
  return angle;
}

double SocialRuleLayer::ComputeSigma(double variance)
{
  // Compute sigma based on the variance
  // Constraint sigma to range [0.2, 0.7] --> Costmap radius will be around [0.5, 2.5]
  double sigma = 0.2 + 0.5 * (variance - 0.25) / (2.0 - 0.25); // variance is radius, range [0.25, 1.0]
  return min(max(sigma, 0.2), 0.7); // Ensure sigma is within [0.2, 0.7]
}

double SocialRuleLayer::Asymmetrical_Gaussian(double x, double y, double x0, double y0, double vx, double vy, double variance) 
{
  Eigen::Vector2d robot_ped_rel_pos(x0 - current_odom_.pose.pose.position.x, 
                                    y0 - current_odom_.pose.pose.position.y);
  Eigen::Vector2d delta(x - x0, y - y0);
  Eigen::Vector2d direction(vx, vy);
  double norm = direction.norm();
  if (norm > 0.2) 
  {
    direction.normalize(); // Normalize the direction vector
    Eigen::Vector2d orthogonal(-direction.y(), direction.x()); // Get the orthogonal vector

    // Project delta onto the direction vector
    double u = delta.dot(direction); // Forward projection
    double v = delta.dot(orthogonal); // Lateral projection

    // Calculate the cost using an asymmetrical Gaussian distribution
    double sigma_front, sigma_back, sigma_right, sigma_left;
    // Adjust the sigma values based on the current rule
    if (current_rule_.rule == "accelerate")
    {
      sigma_front = 0.4;//sqrt(variance * 7.5) / 10.0; // Less spread in the forward direction
      sigma_back = 0.2;//sqrt(variance * 5.0) / 10.0; 
      sigma_right = 0.2;//sqrt(variance * 10.0) / 10.0; // Rightward spread
      sigma_left = 0.2;//sqrt(variance * 10.0) / 10.0; // Leftward spread
    }
    else if(current_rule_.rule == "decelerate")
    {
      sigma_front = 1.0;//sqrt(variance * 20.0) / 10.0; // More spread in the forward direction
      sigma_back = 0.2;//sqrt(variance * 7.5) / 10.0; // More spread in the backward direction
      sigma_right = 0.4;//sqrt(variance * 15.0) / 10.0; // Rightward spread
      sigma_left = 0.4;//sqrt(variance * 15.0) / 10.0; // Leftward spread        
    }
    else if(current_rule_.rule == "turn_left")
    {
      sigma_front = 1.0;//sqrt(variance * 10.0) / 10.0; // Frontward spread
      sigma_back = 0.2;//sqrt(variance * 5.0) / 10.0; // Backward spread
      sigma_right = 0.5;//sqrt(variance * 20.0) / 10.0; // Rightward spread
      sigma_left = 0.2;//sqrt(variance * 5.0) / 10.0; // Leftward spread        
    }
    else if(current_rule_.rule == "turn_right")
    {
      sigma_front = 1.0;//sqrt(variance * 10.0) / 10.0; // Frontward spread
      sigma_back = 0.2;//sqrt(variance * 5.0) / 10.0; // Backward spread
      sigma_right = 0.2;//sqrt(variance * 5.0) / 10.0; // Rightward spread
      sigma_left = 0.5;//sqrt(variance * 20.0) / 10.0; // Leftward spread        
    }
    else // "normal" or other rules
    {
      sigma_front = 1.0;//sqrt(variance * 10.0) / 10.0; // Frontward spread
      sigma_back = 0.2;//sqrt(variance * 5.0) / 10.0; // Backward spread
      sigma_right = 0.2;//sqrt(variance * 10.0) / 10.0; // Rightward spread
      sigma_left = 0.2;//sqrt(variance * 10.0) / 10.0; // Leftward spread
    }

    double sigma_u = (u >= 0) ? sigma_front : sigma_back; // Use different sigma for forward/backward
    double sigma_v = (v >= 0) ? sigma_left : sigma_right; // Use different sigma for left/right
    if (robot_ped_rel_pos.dot(direction) < 0) // If the pedestrian is towards the robot
    {
      sigma_v = (v >= 0) ? sigma_right : sigma_left;
    }

    double exponent = - (u * u) / (2 * sigma_u * sigma_u) - (v * v) / (2 * sigma_v * sigma_v);
    double amplitude = 254.0; // Maximum cost value (leave some space for lethal cost)
    double cost = amplitude * exp(exponent);
    // Limit cost value to the range [0, 254]
    return min(max(cost, 0.0), 254.0);
  }
  else
  {
    // Velocity is too small, return a standard gaussian cost centered at (x0, y0)
    double sigma = 0.2;//ComputeSigma(variance); // Standard deviation
    double exponent = - ((x - x0) * (x - x0) + (y - y0) * (y - y0)) / (2 * sigma * sigma);
    double amplitude = 254.0; // Maximum cost value (leave some space for lethal cost)
    double cost = amplitude * exp(exponent);
    // Limit cost value to the range [0, 254]
    return min(max(cost, 0.0), 254.0);
  }
}

} // namespace social_rules_selector

// Register this plugin with pluginlib
PLUGINLIB_EXPORT_CLASS(social_rule_layer::SocialRuleLayer, costmap_2d::Layer)