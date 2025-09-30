#!/bin/bash

# 設定 TurtleBot3 的型號
export TURTLEBOT3_MODEL=waffle_pi
if [[ "$TURTLEBOT3_MODEL" == "waffle_pi" ]]; then
  echo "✅ TURTLEBOT3_MODEL set to '$TURTLEBOT3_MODEL' (成功)"
else
  echo "❌ 無法設定 TURTLEBOT3_MODEL" >&2
  exit 1
fi

# 檢查工作目錄下是否有 devel/setup.bash
if [[ ! -f "devel/setup.bash" ]]; then
  echo "❌ 找不到 devel/setup.bash，請先確認是否已經編譯過工作區（catkin_make 或 catkin build）" >&2
  exit 1
fi

# Source ROS 工作區
if source devel/setup.bash; then
  echo "✅ ROS workspace sourced successfully (成功)"
else
  echo "❌ ROS workspace source 失敗" >&2
  exit 1
fi

# 額外檢查：確認基本的 ROS 環境變數已經被設定
if [[ -z "$ROS_PACKAGE_PATH" ]]; then
  echo "❌ ROS_PACKAGE_PATH 為空，可能是 source 沒有生效" >&2
  exit 1
else
  echo "✅ ROS_PACKAGE_PATH = $ROS_PACKAGE_PATH"
fi

echo "🎉 全部步驟完成 (Setup completed successfully!)"
