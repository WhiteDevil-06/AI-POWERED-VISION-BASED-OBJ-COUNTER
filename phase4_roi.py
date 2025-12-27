import cv2
import time
import numpy as np
from ultralytics import YOLO
from collections import OrderedDict
from scipy.spatial import distance as dist

# --- IMPROVED TRACKER (Tuned for Stability) ---
class CentroidTracker:
    def __init__(self, max_disappeared=20, max_distance=80):
        # OPTIMIZED FOR LOW FPS (5 FPS):
        # max_disappeared=20: If we miss it for ~4 sec (at 5fps), drop it.
        # max_distance=80: Objects move 'further' per frame at low FPS, so we search wider.
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

def phase4_roi():
    print("--- PHASE 4: ROI COUNTING (REFINED) ---")
    
    # 1. Warmup Logic
    print("Initializing logic with 3s warmup...")
    start_warmup = time.time()
    
    # Load Custom Model (V2 Clean) if available
    try:
        model = YOLO("best_v2.pt")
        print("✅ SUCCESS: Found Custom Model 'best_v2.pt' (Clean Version)")
        # Custom model only has 1 class (Index 0: cardboard_box)
        ALLOWED_CLASSES = {0} 
    except:
        print("⚠️ WARNING: Custom Model not found. Using Standard YOLOv8n.")
        model = YOLO("yolov8n.pt")
        # Standard COCO Classes (Backpack, Umbrella, Handbag, Bottle, Laptop, Cell phone, Book)
        ALLOWED_CLASSES = {24, 26, 28, 39, 63, 67, 73}

    # RE-INITIALIZE TRACKER & VARIABLES (Restoring missing code)
    ct = CentroidTracker(max_disappeared=50, max_distance=100)
    roi_box = (150, 100, 340, 280) # x, y, w, h
    total_counts = 0
    counted_ids = set()
    stuck_ids = {} 
    STUCK_THRESHOLD = 5.0 

    # Fix for Error -1072875772: Use DirectShow (CAP_DSHOW) on Windows
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3, 640)
    cap.set(4, 480)

    print("Running...")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Warmup Check
        if time.time() - start_warmup < 3:
            cv2.putText(frame, f"WARMUP: {3 - int(time.time() - start_warmup)}", (250, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Phase 4: ROI Counting", frame)
            cv2.waitKey(1)
            continue

        # Detection
        # Detection
        # V2 Model Tuning: 0.65 is a good balance for high-quality data
        results = model(frame, verbose=False, conf=0.65)
        rects = []
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) in ALLOWED_CLASSES:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype("int")
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    rects.append((x1, y1, x2, y2, conf, cls_id))
        
        # Prepare pure bounding boxes for Tracker
        pure_rects = [r[:4] for r in rects]
        objects = ct.update(pure_rects)
        
        # Draw ROI
        rx, ry, rw, rh = roi_box
        cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), (0, 255, 255), 2)
        cv2.putText(frame, "INTAKE BOX", (rx, ry-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Draw ALL Detections with Debug Info
        for (x1, y1, x2, y2, conf, cls_id) in rects:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Label + Percentage (e.g. "Box: 88%")
            try:
                name = model.names[cls_id]
            except:
                name = str(cls_id)
            
            label = f"{name}: {int(conf * 100)}%"
            
            # Green Background for text (High Contrast)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + t_size[0], y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        current_time = time.time()
        
        for (object_id, centroid) in objects.items():
            cx, cy = centroid
            
            # Logic: Center Bias to prevent edge counting
            # Object must be INSIDE, with a margin of 10px from edge
            in_roi = (rx + 10 < cx < rx + rw - 10) and (ry + 10 < cy < ry + rh - 10)
            
            if in_roi:
                # Count
                if object_id not in counted_ids:
                    total_counts += 1
                    counted_ids.add(object_id)
                    cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), (0, 255, 0), 4) # Flash Green
                
                # Stuck
                if object_id not in stuck_ids:
                    stuck_ids[object_id] = current_time
                elif (current_time - stuck_ids[object_id]) > STUCK_THRESHOLD:
                    cv2.putText(frame, "WARNING: STUCK!", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            else:
                if object_id in stuck_ids:
                    del stuck_ids[object_id]

            # Visuals
            text = f"ID {object_id}"
            color = (0, 255, 0) if object_id in counted_ids else (0, 0, 255)
            cv2.putText(frame, text, (cx - 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.circle(frame, (cx, cy), 4, color, -1)

        # Cleanup
        for sid in list(stuck_ids.keys()):
            if sid not in objects:
                del stuck_ids[sid]

        cv2.rectangle(frame, (0, 0), (640, 50), (0, 0, 0), -1)
        cv2.putText(frame, f"Boxes Counted: {total_counts}", (10, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Phase 4: ROI Counting", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Final Count: {total_counts}")

if __name__ == "__main__":
    phase4_roi()
