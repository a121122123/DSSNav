# Message containing the predicted trajectories of a tracked group
#
uint64      group_id        # unique identifier of the target, consistent over time
geometry_msgs/PoseWithCovariance predicted_center  # 預測後的群組中心（可含不確定性）
geometry_msgs/Twist predicted_velocity         # 預測後的群組速度
geometry_msgs/PoseStamped[] predicted_trajectory  # 預測出的軌跡 (以多個 PoseStamped 表示)
float64[] prediction_params                    # 預測模型參數，例如：[v_x, v_y, 0.0]，常速模型下假設加速度為0
float64 prediction_score                       # 預測信心分數，預設1.0，日後可依實際情況調整
