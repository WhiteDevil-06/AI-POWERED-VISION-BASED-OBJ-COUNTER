import cv2
import numpy as np
import time
from collections import OrderedDict
from scipy.spatial import distance as dist
from ultralytics import YOLO

# --- CENTROID TRACKER CLASS ---
class CentroidTracker:
    def __init__(self, max_disappeared=40, max_distance=50):
        self.next_object_id = 0
        self.objects = OrderedDict() # ID -> Centroid
        self.disappeared = OrderedDict() # ID -> Disappeared Count
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
        # rects = list of [x1, y1, x2, y2]
        
        # 1. If no detections, increment disappeared
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        # 2. Calculate centroids
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            input_centroids[i] = (cX, cY)

        # 3. If no objects, register all
        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i])
        
        # 4. Match existing objects
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            D = dist.cdist(np.array(object_centroids), input_centroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row][col] > self.max_distance:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            # Deregister disappeared
            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            # Register new
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)
            for col in unused_cols:
                self.register(input_centroids[col])

        return self.objects

# --- MAIN PHASE 3 SCRIPT ---
def phase3_tracking():
    print("--- PHASE 3: TRACKING & ID STABILITY ---")
    
    # 1. Load Model
    print("Loading YOLOv8n...")
    model = YOLO("yolov8n.pt")
    
    # 2. Init Tracker
    ct = CentroidTracker(max_disappeared=30, max_distance=60)
    
    # Allowed: Carton Proxies
    ALLOWED_CLASSES = {24, 26, 28, 39, 63, 67, 73} 

    # 3. Camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Success. Running Loop with IDs...")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Inference
        results = model(frame, verbose=False, conf=0.55)
        
        # Prepare Rects for Tracker
        rects = []
        
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id in ALLOWED_CLASSES:
                    rects.append(box.xyxy[0].cpu().numpy().astype("int"))

        # Update Tracker
        objects = ct.update(rects)
        
        # Visualize
        # Draw Bounding Boxes (Green)
        for (x1, y1, x2, y2) in rects:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # Draw IDs (Red)
        for (object_id, centroid) in objects.items():
            text = f"ID {object_id}"
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 0, 255), -1)

        cv2.putText(frame, f"PHASE 3: TRACKING | IDs: {len(objects)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Phase 3: IDs", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Phase 3 Complete.")

if __name__ == "__main__":
    phase3_tracking()
