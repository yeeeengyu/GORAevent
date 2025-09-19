import cv2
import time
import random
import numpy as np
from ultralytics import YOLO
from modules.centerdist import centerdist
from modules.clamp import clampbox
from modules.iou import iou
from modules.particles import init_particle, update_particle

# 파티클변수 ( 고치고싶으면 고치기 )
NUM_PARTICLES = 60
BASE_VY = (0.01, 0.03)
WIND_VX = (-0.004, 0.004)
GRAVITY = 0.0006
RADIUS_PX = (2, 4)
ALPHA = 0.55
PRUNE = 15
prev = time.time()
fps = 0.0
color_choices = [(0,255,255), (255,200,0), (255,105,180), (180,105,255), (200,255,200)]
model = YOLO("camera\\models\\yolo11n.pt")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("딴카메라")
particles_by_id = {}
prev_box_by_id = {}
last_seen_frames = {}
next_id = 0
frame_idx = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    flipped = cv2.flip(frame, 1) # 화면좌우반전
    H, W = flipped.shape[:2]
    results = model(flipped, imgsz=480, conf=0.7)[0] # fps 안나오면 imgsz 낮추기
    anot = results.plot()
    det_boxes = []
    if results.boxes is not None and len(results.boxes) > 0:
        bxs = results.boxes.xyxy.cpu().numpy()
        for b in bxs:
            x1, y1, x2, y2 = clampbox(*b, W, H)
            if x2 > x1 and y2 > y1:
                det_boxes.append([x1,y1,x2,y2])
    assigned = {}
    used_ids = set()
    IOU_TH = 0.3

    for db in det_boxes:
        best_id, best_score = None, -1.0
        for fid, prev_box in prev_box_by_id.items():
            if fid in used_ids:
                continue
            if prev_box is None:
                continue
            score = iou(db, prev_box)
            if score < IOU_TH:
                d = centerdist(db, prev_box)
                w = db[2]-db[0]; h = db[3]-db[1]
                diag = (w*w + h*h) ** 0.5 + 1e-6
                score = 1.0 - min(1.0, d/diag) * 0.9
            if score > best_score:
                best_score = score
                best_id = fid
        if best_id is None or best_score < 0.25:
            best_id = next_id
            next_id += 1
            particles_by_id[best_id] = init_particle()
            prev_box_by_id[best_id] = None

        assigned[best_id] = db
        used_ids.add(best_id)

    for fid in list(particles_by_id.keys()):
        if fid not in assigned and (frame_idx - last_seen_frames.get(fid, -1) > PRUNE):
            particles_by_id.pop(fid, None)
            prev_box_by_id.pop(fid, None)
            last_seen_frames.pop(fid, None)

    overlay = anot.copy()
    for fid, box in assigned.items():
        last_seen_frames[fid] = frame_idx
        x1, y1, x2, y2 = box
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)

        ps = particles_by_id[fid]
        update_particle(ps)
        for (u, v, vu, vv, r) in ps:
            px = int(x1 + u * bw)
            py = int(y1 + v * bh)
            if x1 <= px < x2 and y1-2 <= py < y2+2:
                cv2.circle(overlay, (px, py), r, random.choice(color_choices), -1, lineType=cv2.LINE_AA)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 200, 255), 2) # 박스지우려면 이거 지우셈
    cv2.addWeighted(overlay, ALPHA, anot, 1-ALPHA, 0, anot)

    for fid, box in assigned.items():
        prev_box_by_id[fid] = box
    now = time.time()
    if now > prev:
        fps = 0.9*fps + 0.1*(1.0/(now - prev))
    prev = now
    cv2.putText(anot, f"FPS: {fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    cv2.imshow("cam", anot)
    frame_idx += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
