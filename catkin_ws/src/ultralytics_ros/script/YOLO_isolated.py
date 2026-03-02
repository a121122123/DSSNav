#!/usr/bin/env python3
# Host-side YOLO GPU inference server
import socket
import cv2
import numpy as np
import json
from ultralytics import YOLO

model = YOLO("../models/best.pt")    # 改成你需要的


server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("0.0.0.0", 5001))
server.listen(1)

print("[HOST] Waiting for docker...")
conn, addr = server.accept()
print("[HOST] Connected:", addr)


while True:
    # 先收 4 bytes 的影像長度
    length_bytes = conn.recv(4)
    if not length_bytes:
        break

    length = int.from_bytes(length_bytes, 'big')

    jpg_bytes = conn.recv(length, socket.MSG_WAITALL)

    # Decode JPEG → cv2 image
    nparr = np.frombuffer(jpg_bytes, np.uint8).reshape((480, 640, 3))
    #frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # YOLO inference
    results = model.track(nparr, persist=True, verbose=False)[0]

    detections = []
    boxes = results.boxes

    for i in range(len(boxes)):
        xywh = boxes.xywh[i].cpu().numpy()
        cls = int(boxes.cls[i])
        conf = float(boxes.conf[i])
        track_id = int(boxes.id[i]) if boxes.id is not None else -1

        detections.append({
            "cx": float(xywh[0]),
            "cy": float(xywh[1]),
            "w": float(xywh[2]),
            "h": float(xywh[3]),
            "cls": results.names[cls],
            "conf": conf,
            "id": track_id
        })

    json_msg = json.dumps({"detections": detections})
    json_bytes = json_msg.encode()
    json_length = len(json_bytes)
    conn.send(json_length.to_bytes(4, 'big'))
    conn.sendall(json_bytes)
