import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation
import numpy as np
import argparse
import os

# =========================================================
# 參數讀取
# =========================================================
parser = argparse.ArgumentParser(description="Visualize robot and pedestrian paths from pickle data.")
parser.add_argument("save_name", type=str, help="Name of the pickle file (without extension)")
parser.add_argument("--path", type=str,
                    default="/home/andre/ros_docker_ws/catkin_ws/src/social_rules_selector",
                    help="Base directory containing pickle files")
args = parser.parse_args()

save_name = args.save_name
save_path = os.path.join(args.path, f"{save_name}.pkl")

# ----- 讀取資料 -----
with open(save_path, "rb") as f:
    data = pickle.load(f)

robot_path = data["robot_path"]
robot_times = data["robot_times"]
robot_rules = data["robot_rules"]
pedestrian_paths = data["pedestrian_paths"]
pedestrian_times = data["pedestrian_times"]
map_data = data["map_data"]
map_resolution = data["map_resolution"]
map_origin = data["map_origin"]

# ----- 顏色 -----
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

# =========================================================
# 1️⃣ 畫靜態圖
# =========================================================
def plot_static():
    fig, ax = plt.subplots(figsize=(10, 10))

    # 地圖
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

    # 機器人路徑（依照 rule 著色）
    if len(robot_path) > 1:
        points = np.array(robot_path)
        segments = np.concatenate([points[:-1, None], points[1:, None]], axis=1)
        colors = [get_rule_color(rule) for rule in robot_rules[:-1]]
        lc = LineCollection(segments, colors=colors, linewidths=2, label="Robot")
        ax.add_collection(lc)
    elif len(robot_path) == 1:
        ax.plot(robot_path[0][0], robot_path[0][1], marker='o', color=get_rule_color(robot_rules[0]), label="Robot")

    # 行人路徑
    cmap_list = ['Reds', 'Greens', 'Purples', 'Oranges', 'YlGn', 'BuPu', 'Greys']
    for idx, (agent_id, path) in enumerate(pedestrian_paths.items()):
        if path:
            px, py = zip(*path)
            color = cm.get_cmap(cmap_list[idx % len(cmap_list)])(0.7)
            ax.plot(px, py, label=f'Ped {agent_id}', color=color, linestyle='--', alpha=0.7)

    ax.set_title("Final Paths Snapshot (from PKL)")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.axis("equal")
    ax.grid()
    ax.legend()

    plt.show()

# =========================================================
# 2️⃣ 動畫
# =========================================================
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
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

    ax.set_title("Animated Robot & Pedestrian Paths")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.grid()

    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                        bbox=dict(facecolor='white', alpha=0.7))

    robot_line, = ax.plot([], [], lw=3, label='Robot')
    ped_lines = {}
    for idx, agent_id in enumerate(pedestrian_paths.keys()):
        line, = ax.plot([], [], lw=2, label=f'Ped {agent_id}', linestyle='--', alpha=0.7)
        ped_lines[agent_id] = line

    all_times = robot_times + sum(pedestrian_times.values(), [])
    if not all_times:
        print("[⚠] No data recorded. Exiting animation.")
        return
    t_min, t_max = min(all_times), max(all_times)
    steps = 300
    time_seq = np.linspace(t_min, t_max, steps)

    def get_current_rule(cur_time):
        idxs = [i for i, t in enumerate(robot_times) if t <= cur_time]
        return robot_rules[idxs[-1]] if idxs else "normal"

    def update(frame):
        cur_time = time_seq[frame]
        cur_rule = get_current_rule(cur_time)
        time_text.set_text(f"Time: {cur_time - t_min:4.1f}s | Rule: {cur_rule}")

        rx = [p[0] for i, p in enumerate(robot_path) if robot_times[i] <= cur_time]
        ry = [p[1] for i, p in enumerate(robot_path) if robot_times[i] <= cur_time]
        robot_line.set_data(rx, ry)
        robot_line.set_color(get_rule_color(cur_rule))

        for agent_id, line in ped_lines.items():
            px = [p[0] for i, p in enumerate(pedestrian_paths[agent_id]) if pedestrian_times[agent_id][i] <= cur_time]
            py = [p[1] for i, p in enumerate(pedestrian_paths[agent_id]) if pedestrian_times[agent_id][i] <= cur_time]
            line.set_data(px, py)

        return [robot_line] + list(ped_lines.values()) + [time_text]

    ani = FuncAnimation(fig, update, frames=range(steps), interval=50, blit=True)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()

    # 直接顯示動畫
    plt.show()

    # 如果想要存檔，可以取消註解：
    # ani.save("ros_path_animation.mp4", writer='ffmpeg', fps=20)

# =========================================================
# 執行
# =========================================================
plot_static()
animate_paths()