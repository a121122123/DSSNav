#!/bin/bash

echo "啟動模擬環境..."

# 啟動 Gazebo
roslaunch pedsim_gazebo_plugin social_contexts.launch &
PID_GAZEBO=$!
sleep 5

# Spawn robot
roslaunch launch/spawn_turtlebot.launch \
  yaw:=0.7854 &
PID_TURTLEBOT=$!
sleep 3

# 啟動 Tracker
roslaunch ultralytics_ros my_tracker_with_cloud.launch \
  yolo_model:=yolov11n_combine.pt \
  debug:=false \
  map_file:=/home/andre/ros_docker_ws/catkin_ws/map/test.yaml \
  initial_pose_a:=0.7854 &
PID_TRACKER=$!
sleep 3

# 啟動 Social Rule Selector
roslaunch social_rules_selector social_rule.launch \
  weight_obs:=10.0 \
  weight_goal:=1.0 \
  weight_path:=0.5 \
  weight_speed:=0.1 \
  weight_ped:=10.0 &
PID_RULE=$!
sleep 3

# 暫停 Gazebo 模擬
echo "暫停 Gazebo 模擬..."
rosservice call /gazebo/pause_physics
sleep 1

# 啟動 PedSim
roslaunch pedsim_simulator simple_pedestrians.launch with_robot:=false &
PID_PEDSIM=$!
sleep 3

# 發送 2D Nav Goal
echo "發送 2D Nav Goal 到機器人..."
rostopic pub /move_base_simple/goal geometry_msgs/PoseStamped \
  '{header: {frame_id: "map"}, pose: {position: {x: 20.0, y: 20.0, z: 0.0}, orientation: {x: 0.0, y: 0.0, z: 0.7071, w: 0.7071}}}' \
  -1

# 啟動紀錄腳本
echo "啟動紀錄腳本..."
rosrun social_rules_selector path_recorder.py &
PID_RECORDER=$!
sleep 3

# 恢復 Gazebo 模擬
echo "恢復 Gazebo 模擬..."
rosservice call /gazebo/unpause_physics

echo "所有組件已啟動。按 Ctrl+C 可結束模擬。"

# 捕捉 Ctrl+C 並停止所有子進程
trap "echo '停止所有程序...'; \
  kill $PID_GAZEBO $PID_TURTLEBOT $PID_TRACKER $PID_RULE $PID_PEDSIM $PID_RECORDER 2>/dev/null; exit" SIGINT

# 永久等待直到被中斷
wait