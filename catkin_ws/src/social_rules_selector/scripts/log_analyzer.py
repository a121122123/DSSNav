import pickle, glob, os, csv
import numpy as np

def analyze_log(path):
    with open(path, "rb") as f:
        data = pickle.load(f)

    robot_path = data["robot_path"]
    robot_times = data["robot_times"]
    pedestrian_paths = data["pedestrian_paths"]
    pedestrian_times = data["pedestrian_times"]

    # 1. Elapsed time
    elapsed = robot_times[-1] - robot_times[0] if robot_times else 0

    # 2. Path length
    if len(robot_path) > 1:
        path_len = np.sum(np.linalg.norm(np.diff(np.array(robot_path), axis=0), axis=1))
    else:
        path_len = 0

    # 3. Min. distance
    min_dist = float("inf")
    for t_idx, (rx, ry) in enumerate(robot_path):
        r_time = robot_times[t_idx]
        for agent_id, ped_path in pedestrian_paths.items():
            ped_times = pedestrian_times[agent_id]
            if not ped_times:
                continue
            nearest_idx = np.argmin([abs(pt - r_time) for pt in ped_times])
            px, py = ped_path[nearest_idx]
            dist = np.hypot(rx - px, ry - py)
            if dist < min_dist:
                min_dist = dist
    if min_dist == float("inf"):
        min_dist = None

    # 4. Failure (rule-based)
    collision = (min_dist is not None and min_dist < 0.5)   # threshold = 0.5m
    timeout   = (elapsed > 60.0)                           # threshold = 60s

    return {
        "file": os.path.basename(path),
        "elapsed": elapsed,
        "path_len": path_len,
        "min_dist": min_dist,
        "collision": collision,
        "timeout": timeout,
    }

if __name__ == "__main__":
    logs = glob.glob(os.path.expanduser("~/ros_docker_ws/catkin_ws/src/social_rules_selector/*.pkl"))
    results = [analyze_log(f) for f in logs]

    if not results:
        print("⚠ No logs found.")
        exit()

    elapsed_avg = np.mean([r["elapsed"] for r in results])
    pathlen_avg = np.mean([r["path_len"] for r in results])
    mindist_avg = np.mean([r["min_dist"] for r in results if r["min_dist"] is not None])

    num_runs = len(results)
    num_collision = sum(r["collision"] for r in results)
    num_timeout = sum(r["timeout"] for r in results)

    # --- Print ---
    print("=== Experiment Summary ===")
    print(f"Total runs: {num_runs}")
    print(f"Avg elapsed time: {elapsed_avg:.2f} s")
    print(f"Avg path length: {pathlen_avg:.2f} m")
    print(f"Avg min distance: {mindist_avg:.2f} m")
    print(f"Failure rate: {100*num_collision/num_runs:.1f}% collision, {100*num_timeout/num_runs:.1f}% timeout")
    print("==========================")

    # --- Save CSV ---
    save_csv = os.path.expanduser("~/ros_docker_ws/catkin_ws/src/social_rules_selector/analysis_results.csv")
    with open(save_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

        # 加 summary row
        writer.writerow({
            "file": "SUMMARY",
            "elapsed": elapsed_avg,
            "path_len": pathlen_avg,
            "min_dist": mindist_avg,
            "collision": f"{100*num_collision/num_runs:.1f}% collision",
            "timeout": f"{100*num_timeout/num_runs:.1f}% timeout",
        })

    print(f"[✔] Detailed results saved to {save_csv}")
