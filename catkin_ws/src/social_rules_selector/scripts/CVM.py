#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, PoseWithCovariance, Twist
# from PredictTraj.msg import PredictTraj
from social_rules_selector.msg import PredictTraj, PredictTrajs
from std_msgs.msg import Header

def predict_trajectory(tracked_groups, predict_time=1.0, dt=0.1):
    """
    對於每個收到的群組，根據其目前的中心位置(centerOfGravity)與速度(group_velocity)
    使用常速模型做外推預測，預測時間為 predict_time，時間間隔為 dt。
    
    輸出一個 PredictTrajs 訊息，其 predictions 陣列中，每一個元素為一個群組的預測結果:
      - group_id         : 該群組的唯一識別碼
      - predicted_center : 外推得到的預測中心（可直接取最後一次外推位置作為代表）
      - predicted_velocity: 預測速度（此處常速模型，與原始速度一致）
      - predicted_trajectory: 一連串的預測位置 (PoseStamped[])
      - prediction_params: 例如 [v_x, v_y, 0.0] (常速模型下的預測參數)
      - prediction_score : 預測信心，這裡暫定為 1.0
    """
    # 從 tracked_groups 中取得 groups 陣列，若為空則回傳空的 PredictTrajs
    if not tracked_groups.groups:
        # rospy.loginfo("received empty tracked_groups, empty prediction will be return.")
        pred_all = PredictTrajs()
        pred_all.header = tracked_groups.header
        pred_all.predicted_trajs = []
        return pred_all

    predictions = []
    num_steps = int(np.ceil(predict_time / dt))
    
    # 對每個群組進行預測
    for group in tracked_groups.groups:
        # 取出群組基本資訊
        group_id = group.group_id
        current_center = group.centerOfGravity  # PoseWithCovariance
        current_twist = group.group_velocity      # Twist
        
        # 建立預測軌跡列表
        trajectory = []
        # 先取得初始位置與假設的常數速度，這裡用速度分量 linear.x 與 linear.y
        init_x = current_center.pose.position.x
        init_y = current_center.pose.position.y
        v_x = current_twist.linear.x
        v_y = current_twist.linear.y
        
        # 根據常速線性外推，計算每個時間點的位置
        for step in range(1, num_steps + 1):
            t_future = step * dt
            x_pred = init_x + v_x * t_future
            y_pred = init_y + v_y * t_future
            
            pred_pose = PoseStamped()
            # pred_pose.header = tracked_groups.header  # 或可使用目前時間戳，但保持一致即可
            # # 預測時間戳為目前時間加上外推的秒數
            # pred_pose.header.stamp = rospy.Time.from_sec(tracked_groups.header.stamp.to_sec() + t_future)
            pred_pose.header = Header()
            pred_pose.header.frame_id = tracked_groups.header.frame_id
            pred_pose.header.stamp = tracked_groups.header.stamp + rospy.Duration.from_sec(t_future)
            pred_pose.pose.position.x = x_pred
            pred_pose.pose.position.y = y_pred
            pred_pose.pose.position.z = current_center.pose.position.z  # 保持原 z 值
            # 預測的朝向可簡單繼承原中心的朝向
            pred_pose.pose.orientation = current_center.pose.orientation
            
            trajectory.append(pred_pose)
        
        # 在常速模型下，預測參數可設定為群組當前速度與假設加速度=0
        prediction_params = [v_x, v_y, 0.0]
        # 預測結果信心先設定為 1.0 (未來可依據量測誤差等進行調整)
        prediction_score = 1.0
        
        # 將本群組的預測結果打包到一個 PredictTraj 訊息中
        group_pred = PredictTraj()
        group_pred.group_id = group_id
        # 使用最後一個預測點作為預測中心
        if trajectory:
            # 將最後一點包裝為 PoseWithCovariance
            predicted_center = PoseWithCovariance()
            predicted_center.pose = trajectory[-1].pose
            # covariance 暫設維持原中心的 covariance
            predicted_center.covariance = list(current_center.covariance)
            group_pred.predicted_center = predicted_center
        else:
            group_pred.predicted_center = current_center
        
        group_pred.predicted_velocity = current_twist
        group_pred.predicted_trajectory = trajectory
        group_pred.prediction_params = prediction_params
        group_pred.prediction_score = prediction_score
        
        predictions.append(group_pred)
    
    # 建構 PredictTrajs 訊息
    pred_all = PredictTrajs()
    pred_all.header = tracked_groups.header
    pred_all.predicted_trajs = predictions
    
    return pred_all

''' For testing purpose only
if __name__ == '__main__':
    # 測試模組：建立一個 dummy 的 TrackedGroups 訊息
    rospy.init_node("predict_trajectory_test")
    from std_msgs.msg import Header
    from geometry_msgs.msg import Pose, PoseWithCovariance, Twist, PoseStamped
    # 注意：請根據你系統中 TrackedGroups.msg 與 TrackedGroup.msg 的定義來建立 dummy 資料
    # 這裡僅作為範例
    dummy_groups = type("DummyTrackedGroups", (), {})()  # 使用簡單對象模擬
    dummy_groups.header = Header()
    dummy_groups.header.stamp = rospy.Time.now()
    # 假設只有一組資料
    try:
        from srl_tracking_msgs.msg import TrackedGroup  # 如果你有此命名空間，可引用；否則用類似結構模擬
    except ImportError:
        # 用簡單的物件模擬
        class DummyTrackedGroup(object):
            pass
        TrackedGroup = DummyTrackedGroup

    group = TrackedGroup()
    group.group_id = 1
    # 建立中心位置與 covariance
    group.centerOfGravity = PoseWithCovariance()
    group.centerOfGravity.pose = Pose()
    group.centerOfGravity.pose.position.x = 1.0
    group.centerOfGravity.pose.position.y = 2.0
    group.centerOfGravity.pose.position.z = 0.0
    group.centerOfGravity.pose.orientation.w = 1.0
    group.centerOfGravity.covariance = [0]*36

    # 建立群組速度（常速）
    group.group_velocity = Twist()
    group.group_velocity.linear.x = 0.5
    group.group_velocity.linear.y = 0.2
    group.group_velocity.linear.z = 0.0

    # track_ids 可忽略
    group.track_ids = [1001, 1002]

    dummy_groups.groups = [group]
    
    pred_msg = predict_trajectory(dummy_groups, predict_time=2.0, dt=0.1)
    rospy.loginfo("預測結果已建立")
    rospy.loginfo(pred_msg)
'''