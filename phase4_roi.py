import cv2
import time
import numpy as np
import os
from ultralytics import YOLO
from collections import OrderedDict
from scipy.spatial import distance as dist

# --- TRACKER CLASS ---
class CentroidTracker:
    def __init__(self, max_disappeared=40, max_distance=100):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            input_centroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            D = dist.cdist(np.array(object_centroids), input_centroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols: continue
                if D[row][col] > self.max_distance: continue
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            unused_cols = set(range(0, D.shape[1])).difference(used_cols)
            for col in unused_cols:
                self.register(input_centroids[col])
        return self.objects

# --- MAIN VISION LOGIC ---
def run_vision_lab():
    print("🚀 LAUNCHING VISION TEST LAB...")
    
    # 1. MODEL LOADING STRATEGY
    model_path = "yolov8n.pt" # Default
    if os.path.exists("mega_v3.pt"):
        model_path = "mega_v3.pt"
        print(f"✅ DETECTED MEGA MODEL: {model_path}")
    elif os.path.exists("roboflow_v2.pt"):
        model_path = "roboflow_v2.pt"
        print(f"✅ DETECTED ROBOFLOW MODEL: {model_path}")
    else:
        print("⚠️ NO CUSTOM MODEL FOUND. USING BASE YOLOV8n.")

    model = YOLO(model_path)
    # Force single class mode if using custom model (Classes are usually 0)
    ALLOWED_CLASSES = [0] if "yolov8n" not in model_path else [0, 39, 41, 67, 73]

    # 2. CAMERA
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # 3. SETTINGS
    roi_rect = (300, 100, 600, 400) # x, y, w, h
    tracker = CentroidTracker()
    
    total_count = 0
    counted_ids = set()
    stuck_log = {}
    
    # 4. LOOP
    print("🟢 SYSTEM LIVE. PRESS 'q' TO EXIT.")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # A. DRAW ROI
        rx, ry, rw, rh = roi_rect
        cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), (0, 255, 0), 2)
        cv2.putText(frame, "ACTIVE ZONE", (rx, ry-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # B. DETECT
        results = model(frame, conf=0.5, verbose=False)
        rects = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id in ALLOWED_CLASSES:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    
                    # Add to tracker
                    rects.append((x1, y1, x2, y2))
                    
                    # Visuals
                    label = f"BOX {int(conf*100)}%"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 2)
                    cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2)

        # C. TRACK & COUNT
        objects = tracker.update(rects)
        
        for (obj_id, centroid) in objects.items():
            cx, cy = centroid
            
            # Check ROI
            in_roi = (rx < cx < rx+rw) and (ry < cy < ry+rh)
            
            if in_roi:
                # Count
                if obj_id not in counted_ids:
                    total_count += 1
                    counted_ids.add(obj_id)
                    # Flash effect
                    cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), (0, 255, 255), 5)
                
                # Stuck Logic
                if obj_id not in stuck_log:
                    stuck_log[obj_id] = time.time()
                elif (time.time() - stuck_log[obj_id]) > 5.0:
                    cv2.putText(frame, "⚠️ STUCK!", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

            else:
                 if obj_id in stuck_log: del stuck_log[obj_id]

            # Draw ID
            cv2.putText(frame, f"ID {obj_id}", (cx-10, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

        # D. HUD
        cv2.putText(frame, f"COUNT: {total_count}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 4)

        cv2.imshow("TEST LAB", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_vision_lab()
