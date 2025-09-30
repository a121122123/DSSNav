#!/bin/bash
# hint : bash start_simulation_hospital.sh [NUM_RUNS] [DEBUG_MODE] [USE_TEB] [ALWAYS_STATIC]
# hint : bash start_simulation_hospital.sh 3 true dwa false

# 取得輸入參數作為迴圈次數，預設 1
NUM_RUNS=${1:-1}
DEBUG_MODE=${2:-false}    # 預設 false
# local_planner: teb or dwa
if [ -z "$3" ]; then
  USE_TEB="true"
else
  USE_TEB=$3
fi
# dynamic social space
if [ -z "$4" ]; then
  ALWAYS_STATIC="false"
else
  ALWAYS_STATIC=$4
fi
echo "NUM_RUNS: $NUM_RUNS, DEBUG_MODE: $DEBUG_MODE, USE_TEB: $USE_TEB, ALWAYS_STATIC: $ALWAYS_STATIC"

sleep 5

echo "===============執行 $NUM_RUNS 次模擬==============="

for i in $(seq 1 $NUM_RUNS)
do

echo "=============== 第 $i 次模擬 ==============="

  # 啟動 Gazebo
  roslaunch launch/gazebo_sim_turtlebot3.launch \
    yaw:=3.14 \
    gui:=$DEBUG_MODE &
  PID_GAZEBO=$!
  sleep 5

  # 啟動 Tracker
  roslaunch ultralytics_ros my_tracker_with_cloud.launch \
    yolo_model:=yolov11n_combine.pt \
    debug:=$DEBUG_MODE \
    initial_pose_a:=3.14 \
    use_teb:=$USE_TEB &
  PID_TRACKER=$!
  sleep 5

  # 啟動 Social Rule Selector
  roslaunch social_rules_selector social_rule.launch \
    always_static:=$ALWAYS_STATIC \
    local_planner:=$( [ "$USE_TEB" = true ] && echo "teb" || echo "dwa" ) &
  PID_RULE=$!
  sleep 5

  # 暫停 Gazebo 模擬
  echo "暫停 Gazebo 模擬..."
  rosservice call /gazebo/pause_physics
  sleep 2

  # 啟動 PedSim
  roslaunch pedsim_simulator  my_pedestrians.launch \
    visualize:=$DEBUG_MODE &
  PID_PEDSIM=$!
  sleep 5

  # 發送 2D Nav Goal
  echo "發送 2D Nav Goal 到機器人..."
  rostopic pub /move_base_simple/goal geometry_msgs/PoseStamped \
    '{header: {frame_id: "map"}, pose: {position: {x: -8.0, y: -21.0, z: 0.0}, orientation: {x: 0.0, y: 0.0, z: 0.9967, w: 0.0802}}}' \
    -1

  # 啟動紀錄腳本
  echo "啟動紀錄腳本..."
  rosrun social_rules_selector path_recorder.py \
    _save_name:=simulation_${i} &
  PID_RECORDER=$!
  sleep 5

  # 恢復 Gazebo 模擬
  echo "恢復 Gazebo 模擬..."
  rosservice call /gazebo/unpause_physics

  # --- 等待目標到達 ---
  rostopic echo -n1 /move_base/result | grep "Goal reached" >/dev/null
  if [ $? -eq 0 ]; then
    echo "目標已到達！"
  fi

  # --- 清理 ---
  echo "清理進程..."
  kill $PID_GAZEBO $PID_TRACKER $PID_RULE $PID_PEDSIM $PID_RECORDER 2>/dev/null
  sleep 30
done

# echo "所有組件已啟動。按 Ctrl+C 可結束模擬。"

# # 捕捉 Ctrl+C 並停止所有子進程
# trap "echo '停止所有程序...'; \
#   kill $PID_GAZEBO $PID_TURTLEBOT $PID_TRACKER $PID_RULE $PID_PEDSIM $PID_RECORDER 2>/dev/null; exit" SIGINT




# # 定義清理函數
# cleanup() {
#   echo -e "\n停止所有程序..."
#   kill $PID_GAZEBO $PID_TRACKER $PID_RULE $PID_PEDSIM $PID_RECORDER 2>/dev/null
#   echo -e "\n模擬環境結束 ✅"
#   exit
# }

# # 捕捉 Ctrl+C 也能正常清理
# trap cleanup SIGINT

# # 用 rostopic 等待 /move_base/result 成功
# rostopic echo -n1 /move_base/result | grep "Goal reached" >/dev/null
# if [ $? -eq 0 ]; then
#   echo "目標已到達！"
#   cleanup
# fi