#!/usr/bin/env python
import rospy
from nav_msgs.msg import Odometry, OccupancyGrid
from pedsim_msgs.msg import AgentStates
from social_rules_selector.msg import SocialRule
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
import numpy as np
import os
import sys
import pickle

robot_path = []
robot_times = []
robot_rules = []  # 對應每筆時間的 social_rule.rule 字串

pedestrian_paths = {}
pedestrian_times = {}

social_rule_current = "normal"  # 初始預設值

map_data = None
map_resolution = 0.0
map_origin = (0.0, 0.0)

# 定義 social_rule 與顏色的映射
RULE_COLOR_MAP = {
    "normal": "blue",
    "accelerate": "green",
    "decelerate": "red",
    "turn_left": "orange",
    "turn_right": "purple",
}
DEFAULT_RULE_COLOR = "gray"

def get_rule_color(rule):
    return RULE_COLOR_MAP.get(rule, DEFAULT_RULE_COLOR)

def robot_callback(msg):
    global social_rule_current
    pos = msg.pose.pose.position
    time = msg.header.stamp.to_sec()
    robot_path.append((pos.x, pos.y))
    robot_times.append(time)
    robot_rules.append(social_rule_current)  # 記錄當前的社會規則字串

def pedestrian_callback(msg):
    time = msg.header.stamp.to_sec()
    for agent in msg.agent_states:
        agent_id = agent.id
        pos = agent.pose.position
        if agent_id not in pedestrian_paths:
            pedestrian_paths[agent_id] = []
            pedestrian_times[agent_id] = []
        pedestrian_paths[agent_id].append((pos.x, pos.y))
        pedestrian_times[agent_id].append(time)

def social_rule_callback(msg):
    global social_rule_current
    social_rule_current = msg.rule

def map_callback(msg):
    global map_data, map_resolution, map_origin
    map_data = np.array(msg.data).reshape((msg.info.height, msg.info.width))
    map_resolution = msg.info.resolution
    map_origin = (msg.info.origin.position.x, msg.info.origin.position.y)

def save_data(save_name):
    save_path = os.path.expanduser(f"~/ros_docker_ws/catkin_ws/src/social_rules_selector/{save_name}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump({
            "robot_path": robot_path,
            "robot_times": robot_times,
            "robot_rules": robot_rules,
            "pedestrian_paths": pedestrian_paths,
            "pedestrian_times": pedestrian_times,
            "map_data": map_data,
            "map_resolution": map_resolution,
            "map_origin": map_origin,
        }, f)
    rospy.loginfo(f"[✔] Data saved to {save_path}")

def save_static_plot(save_name):
    fig, ax = plt.subplots(figsize=(10, 10))

    if map_data is not None:
        display_map = np.zeros_like(map_data, dtype=np.uint8)
        display_map[map_data == 0] = 255
        display_map[map_data == 100] = 0
        display_map[map_data == -1] = 205
        extent = [map_origin[0],
                  map_origin[0] + map_data.shape[1]*map_resolution,
                  map_origin[1],
                  map_origin[1] + map_data.shape[0]*map_resolution]
        ax.imshow(display_map, origin='lower', cmap='gray', extent=extent)

    # 機器人軌跡用 LineCollection 依 rule 分段著色
    if len(robot_path) > 1:
        points = np.array(robot_path)
        segments = np.concatenate([points[:-1, None], points[1:, None]], axis=1)
        # 取得每一段對應的rule顏色（以起點時間對應）
        colors = [get_rule_color(rule) for rule in robot_rules[:-1]]
        lc = LineCollection(segments, colors=colors, linewidths=2, label='Robot')
        ax.add_collection(lc)
    elif len(robot_path) == 1:
        ax.plot(robot_path[0][0], robot_path[0][1], marker='o', color=get_rule_color(robot_rules[0]), label='Robot')

    # 行人路徑，單色
    cmap_list = ['Reds', 'Greens', 'Purples', 'Oranges', 'YlGn', 'BuPu', 'Greys']
    for idx, (agent_id, path) in enumerate(pedestrian_paths.items()):
        if path:
            px, py = zip(*path)
            color = cm.get_cmap(cmap_list[idx % len(cmap_list)])(0.7)
            ax.plot(px, py, label=f'Ped {agent_id}', color=color, linestyle='--', alpha=0.7)

    ax.set_title("Final Paths Snapshot")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.axis("equal")
    ax.grid()
    ax.legend()

    save_path = os.path.expanduser(f"~/ros_docker_ws/catkin_ws/src/social_rules_selector/{save_name}.png")
    plt.savefig(save_path)
    plt.close()
    rospy.loginfo(f"[✔] Static path image saved to: {save_path}")

def animate_paths():
    fig, ax = plt.subplots(figsize=(10, 10))

    if map_data is not None:
        display_map = np.zeros_like(map_data, dtype=np.uint8)
        display_map[map_data == 0] = 255
        display_map[map_data == 100] = 0
        display_map[map_data == -1] = 205
        extent = [map_origin[0],
                  map_origin[0] + map_data.shape[1]*map_resolution,
                  map_origin[1],
                  map_origin[1] + map_data.shape[0]*map_resolution]
        ax.imshow(display_map, origin='lower', cmap='gray', extent=extent)
    else:
        all_x = [p[0] for p in robot_path] + [p[0] for paths in pedestrian_paths.values() for p in paths]
        all_y = [p[1] for p in robot_path] + [p[1] for paths in pedestrian_paths.values() for p in paths]
        margin = 5.0
        print(f"Max x: {max(all_x)}, Min x: {min(all_x)}")
        print(f"Max y: {max(all_y)}, Min y: {min(all_y)}")
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

    ax.set_title("Animated Robot & Pedestrian Paths")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    # ax.axis('equal')
    ax.grid()

    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                        bbox=dict(facecolor='white', alpha=0.7))

    robot_line, = ax.plot([], [], lw=3, label='Robot')
    ped_lines = {}
    cmap_list = ['Reds', 'Greens', 'Purples', 'Oranges', 'YlGn', 'BuPu', 'Greys']

    for idx, agent_id in enumerate(pedestrian_paths.keys()):
        line, = ax.plot([], [], lw=2, label=f'Ped {agent_id}', linestyle='--', alpha=0.7)
        ped_lines[agent_id] = line

    all_times = robot_times + sum(pedestrian_times.values(), [])
    if not all_times:
        rospy.logwarn("No data recorded. Exiting.")
        return
    t_min, t_max = min(all_times), max(all_times)
    steps = 300
    time_seq = np.linspace(t_min, t_max, steps)

    def get_current_rule(cur_time):
        # 找出最新時間點 <= cur_time 的 rule，若無回傳 default
        idxs = [i for i, t in enumerate(robot_times) if t <= cur_time]
        if idxs:
            return robot_rules[idxs[-1]]
        else:
            print(f"[⚠] No rule found for time {cur_time}. Using default 'normal'.")
            return "normal"

    def update(frame):
        cur_time = time_seq[frame]
        cur_rule = get_current_rule(cur_time)
        time_text.set_text(f"Time: {cur_time - t_min:4.1f}s | Rule: {cur_rule}")

        # Robot path up to current time
        rx = [p[0] for i, p in enumerate(robot_path) if robot_times[i] <= cur_time]
        ry = [p[1] for i, p in enumerate(robot_path) if robot_times[i] <= cur_time]

        # 將 robot_line 分段上色 (用LineCollection會比較複雜，這邊動畫用單色即可)
        robot_line.set_data(rx, ry)
        robot_line.set_color(get_rule_color(cur_rule))

        # 行人路徑
        for agent_id, line in ped_lines.items():
            px = [p[0] for i, p in enumerate(pedestrian_paths[agent_id]) if pedestrian_times[agent_id][i] <= cur_time]
            py = [p[1] for i, p in enumerate(pedestrian_paths[agent_id]) if pedestrian_times[agent_id][i] <= cur_time]
            line.set_data(px, py)

        return [robot_line] + list(ped_lines.values()) + [time_text]

    ani = FuncAnimation(fig, update, frames=range(steps), interval=50, blit=True)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()

    # 儲存 mp4 動畫
    save_path = os.path.expanduser("~/ros_docker_ws/catkin_ws/src/social_rules_selector/ros_path_animation.mp4")
    rospy.loginfo(f"[🎞] Saving animation to {save_path} ... (this may take a few seconds)")
    ani.save(save_path, writer='ffmpeg', fps=20)
    rospy.loginfo(f"[✔] Animation saved to {save_path}")

    # # 儲存 gif 動畫
    # gif_path = os.path.expanduser("~/ros_docker_ws/catkin_ws/src/social_rules_selector/ros_path_animation.gif")
    # rospy.loginfo(f"[🎞] Saving animation to {gif_path} ...")
    # ani.save(gif_path, writer=PillowWriter(fps=20))
    # rospy.loginfo(f"[✔] GIF saved to {gif_path}")

    # plt.show()

def main():
    rospy.init_node('path_recorder_and_player')

    save_name = rospy.get_param("~save_name", "ros_path_plot")
    rospy.loginfo(f"Path recorder will save files with base name: {save_name}")

    rospy.Subscriber('/odom', Odometry, robot_callback)
    rospy.Subscriber('/pedsim_simulator/simulated_agents', AgentStates, pedestrian_callback)
    rospy.Subscriber('/social_rule', SocialRule, social_rule_callback)
    # rospy.Subscriber('/map', OccupancyGrid, map_callback)

    rospy.loginfo("Recording paths... Press Ctrl+C to stop and show results.")
    rospy.spin()

    save_data(save_name)
    rospy.loginfo("Path recording complete.")

    # save_static_plot(save_name)
    # animate_paths()
    # rospy.loginfo("Static plot and animation generated.")

    sys.exit(0)  # ← 執行完動畫後自動結束程式

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass





# Without the map, the code will still record the robot and pedestrian paths, but it will not plot them on a map background.
# The paths will still be saved in the `robot_path` and `pedestrian_paths` variables, and the plot will only show the paths without any map context.
# This allows you to visualize the paths even if the map data is not available, but the context of where the paths are in the environment will be missing.
# You can still run the code and it will function correctly, just without the map overlay.
# If you want to run the code without the map, you can comment out the `map_callback` and related plotting code.
# This way, the code will still record the paths and plot them without requiring the map data.
# Or you can run the below code to plot the paths without the map:
'''#!/usr/bin/env python
import rospy
from nav_msgs.msg import Odometry
from pedsim_msgs.msg import AgentStates
from geometry_msgs.msg import PoseStamped
import matplotlib.pyplot as plt
import os

robot_path = []
pedestrian_paths = {}  # key: agent_id, value: list of (x, y)

def robot_callback(msg):
    pos = msg.pose.pose.position
    robot_path.append((pos.x, pos.y))

def pedestrian_callback(msg):
    for agent in msg.agent_states:
        agent_id = agent.id
        pos = agent.pose.position
        if agent_id not in pedestrian_paths:
            pedestrian_paths[agent_id] = []
        pedestrian_paths[agent_id].append((pos.x, pos.y))

def main():
    rospy.init_node('path_recorder', anonymous=True)

    rospy.Subscriber('/odom', Odometry, robot_callback)
    rospy.Subscriber('/pedsim_simulator/simulated_agents', AgentStates, pedestrian_callback)

    rospy.loginfo("Recording paths... Ctrl+C to stop and save plot.")
    rospy.spin()

    # Save and plot
    plot_paths(robot_path, pedestrian_paths)

def plot_paths(robot_path, pedestrian_paths):
    plt.figure(figsize=(10, 10))
    robot_xs, robot_ys = zip(*robot_path)
    plt.plot(robot_xs, robot_ys, label="Robot", linewidth=2, color='blue')

    for agent_id, path in pedestrian_paths.items():
        xs, ys = zip(*path)
        plt.plot(xs, ys, label=f"Pedestrian {agent_id}", linestyle='--')

    plt.title("Robot and Pedestrian Paths")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.legend()
    plt.grid()
    plt.axis("equal")

    save_path = os.path.expanduser("~/ros_docker_ws/catkin_ws/src/social_rules_selector/ros_path_plot.png")
    plt.savefig(save_path)
    plt.show()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass'''

