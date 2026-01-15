import cv2
import time
import numpy as np
import os
import datetime
from collections import OrderedDict

# --- FIX MATPLOTLIB HANG (Must be before importing ultralytics) ---
import matplotlib
matplotlib.use('Agg') 
from ultralytics import YOLO

# --- TRACKER CLASS (PURE NUMPY) ---
class CentroidTracker:
    def __init__(self, max_disappeared=80, max_distance=250): # TUNED: Increased tolerance to prevent ID switching
        self.next_object_id = 1
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
            
            # CALC DISTANCE (PURE NUMPY)
            D = np.zeros((len(object_centroids), len(input_centroids)))
            for i in range(len(object_centroids)):
                for j in range(len(input_centroids)):
                    dx = object_centroids[i][0] - input_centroids[j][0]
                    dy = object_centroids[i][1] - input_centroids[j][1]
                    D[i, j] = np.sqrt(dx*dx + dy*dy)

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

# --- MAIN LOGIC ---
def run_vision_lab():
    print("🚀 STEP 1: LOADING YOLO MODEL (Please Wait 10s)...", flush=True)
    
    # 1. LOAD MODEL (BLOCKING - NO ERROR HIDING)
    path = "roboflow_v2.pt" if os.path.exists("roboflow_v2.pt") else "yolov8n.pt"
    print(f"   -> Found Model: {path}", flush=True)
    
    try:
        # FIX FOR PYTORCH 2.6+ SECURITY ERROR (Monkey Patch)
        import torch
        _orig_load = torch.load
        def safe_load(*args, **kwargs):
            if 'weights_only' not in kwargs: kwargs['weights_only'] = False
            return _orig_load(*args, **kwargs)
        torch.load = safe_load
        
        print("🔒 PATCH APPLIED: Legacy Model Loading Enabled.", flush=True)

        model = YOLO(path)
        
        # Restore original (optional, but good practice)
        torch.load = _orig_load
        
        # Warmup
        model(np.zeros((100,100,3), dtype='uint8'), verbose=False)
        print("✅ STEP 1 COMPLETE: Model Loaded.", flush=True)
    except Exception as e:
        print(f"❌ CRITICAL ERROR LOADING MODEL: {e}")
        return

    # 2. LOAD CAMERA
    print("🚀 STEP 2: CONNECTING TO CAMERA...", flush=True)
    # Try Index 0 first
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened() or not cap.read()[0]:
        print("⚠️ Warning: Camera 0 failed. Trying Index 1...", flush=True)
        cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("❌ CRITICAL ERROR: NO CAMERA FOUND.")
        print("   -> Check if Zoom/Teams/Browser is using it.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("✅ STEP 2 COMPLETE: Camera Active.", flush=True)

    # 3. SETUP LOOP
    roi_rect = (400, 150, 480, 400) # (x, y, w, h) - Make sure this covers the belt
    tracker = CentroidTracker(max_disappeared=80, max_distance=250)
    
    total_count = 0
    expected_count = 100
    counted_ids = set()
    stuck_log = {}
    
    system_active = False 
    start_time = time.time()
    
    print("🟢 SYSTEM LIVE. WINDOW SHOULD BE OPEN.", flush=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ ERROR: Camera stream lost.")
            break
        
        # A. ROI VISUALS
        rx, ry, rw, rh = roi_rect
        roi_color = (0, 255, 0)
        
        # B. DETECT
        detections = []
        cls_filter = [0, 39, 41]
        
        # Run Inference
        results = model(frame, conf=0.5, verbose=False)
        
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id in cls_filter:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0]) # Get Confidence
                    detections.append((x1, y1, x2, y2))
                    
                    # Draw Raw Detection (Dim Color)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 1)
                    # SHOW ACCURACY (USER REQ)
                    cv2.putText(frame, f"{int(conf*100)}%", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)

        # C. SMART START
        if not system_active:
            if len(detections) > 0:
                system_active = True
                print("⚡ SYSTEM ACTIVATED: First Box Detected!")
            elif (time.time() - start_time) > 60:
                system_active = True
                print("⚡ SYSTEM ACTIVATED: Timeout (60s).")

        # D. TRACKING
        if system_active:
            objects = tracker.update(detections)
            ids_to_kill = [] # Fix Loop Mutation

            # Use list() to create a copy of items for iteration
            for (obj_id, centroid) in list(objects.items()):
                cx, cy = centroid
                
                in_roi_x = rx < cx < (rx + rw)
                in_roi_y = ry < cy < (ry + rh)
                in_roi = in_roi_x and in_roi_y
                
                text_color = (0, 255, 0) if in_roi else (0, 100, 255)
                cv2.putText(frame, f"ID {obj_id}", (cx-10, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
                cv2.circle(frame, (cx, cy), 4, text_color, -1)

                if in_roi:
                    if obj_id not in counted_ids:
                        total_count += 1
                        counted_ids.add(obj_id)
                        cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), (0, 255, 255), 5)
                    
                    if obj_id not in stuck_log:
                        stuck_log[obj_id] = time.time()
                    elif (time.time() - stuck_log[obj_id]) > 5.0:
                        # Refined Notification (Bottom of ROI)
                        msg = f"NOTIFICATION: OBJECT {obj_id} STUCK"
                        cv2.putText(frame, msg, (rx, ry + rh + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), (0, 0, 255), 3) # Red border on ROI
                else:
                    if obj_id in stuck_log: del stuck_log[obj_id]
                    # KILL IF EXITING RIGHT (and buffer zone 50px)
                    if cx > (rx + rw + 50): 
                         ids_to_kill.append(obj_id)

            # Cleanup IDs safely
            for kid in ids_to_kill:
                 tracker.deregister(kid)

        # E. HUD
        status_txt = "ACTIVE" if system_active else "WAITING..."
        cv2.putText(frame, f"STATUS: {status_txt}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"EXPECTED: {expected_count}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"ACTUAL:   {total_count}", (30, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, ts, (1100, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        
        roi_thick = 3 if system_active else 1
        cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), roi_color, roi_thick)
        cv2.putText(frame, "SCAN ZONE", (rx, ry-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, roi_color, 1)

        cv2.imshow("Box Sense AI cam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_vision_lab()
