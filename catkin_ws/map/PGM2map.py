import cv2
import numpy as np
import yaml

# 讀取 .yaml 取得解析度與原點
with open("/home/andre/ros_docker_ws/catkin_ws/map/map.yaml", 'r') as f:
    map_info = yaml.safe_load(f)

resolution = map_info['resolution']
origin = map_info['origin']  # [x, y, theta]

# 讀取 .pgm 地圖
map_img = cv2.imread("/home/andre/ros_docker_ws/catkin_ws/map/map.pgm", cv2.IMREAD_GRAYSCALE)

# # 顯示原始地圖
# cv2.imshow("Original Map", map_img)

# 二值化障礙物
_, binary = cv2.threshold(map_img, 50, 255, cv2.THRESH_BINARY_INV)

# # 顯示障礙物遮罩
# cv2.imshow("Binary Obstacles", binary)

# 偵測輪廓 (ALL PARTS)
contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# # === 在複製地圖上畫出偵測到的牆壁線段 ===
# map_with_contours = cv2.cvtColor(map_img, cv2.COLOR_GRAY2BGR)
# for contour in contours:
#     approx = cv2.approxPolyDP(contour, 2, True)
#     for i in range(len(approx)):
#         p1 = approx[i][0]
#         p2 = approx[(i + 1) % len(approx)][0]
#         cv2.line(map_with_contours, tuple(p1), tuple(p2), (0, 0, 255), 1)

# # 顯示帶有障礙線段的地圖
# cv2.imshow("Map with Obstacles", map_with_contours)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 轉換為世界座標並輸出 obstacle
for contour in contours:
    approx = cv2.approxPolyDP(contour, 2, True)
    for i in range(len(approx)):
        p1 = approx[i][0]
        p2 = approx[(i+1)%len(approx)][0]

        # 像素轉世界座標：x = origin_x + col * resolution
        x1 = origin[0] + p1[0] * resolution
        y1 = origin[1] + (map_img.shape[0] - p1[1]) * resolution + 1
        x2 = origin[0] + p2[0] * resolution
        y2 = origin[1] + (map_img.shape[0] - p2[1]) * resolution + 1

        print(f'<obstacle x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}"/>')
