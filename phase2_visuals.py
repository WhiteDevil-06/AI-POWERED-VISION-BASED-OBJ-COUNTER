import cv2
import time
import logging
from ultralytics import YOLO

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Phase2")

def phase2_visuals():
    print("--- PHASE 2: BOUNDING BOX VISUALIZATION ---")
    
    # 1. Load Model
    print("Loading YOLOv8n...")
    model = YOLO("yolov8n.pt")
    
    # 2. Define Allowed Classes (Carton Proxies)
    # COCO Indices:
    # 24: backpack, 26: handbag, 28: suitcase
    # 39: bottle, 63: laptop, 67: cell phone, 73: book
    ALLOWED_CLASSES = {24, 26, 28, 39, 63, 67, 73} 
    CLASS_NAMES = model.names

    # 3. Init Camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Camera not accessible.")
        return

    print("Success. Running Loop...")
    print("Look for: Backpack, Suitcase, Laptop, Book, Cell Phone (Proxies for Cartons)")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Inference
        results = model(frame, verbose=False, conf=0.55)
        
        # Process Detections
        det_count = 0
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                # FILTER: Only allowed classes
                if cls_id in ALLOWED_CLASSES:
                    det_count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = f"{CLASS_NAMES[cls_id]} {conf:.2f}"
                    
                    # Draw Box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw Label Background
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
                    cv2.putText(frame, label, (x1, y1 - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # UI Overlay
        cv2.putText(frame, f"PHASE 2: VISUALIZATION | Detections: {det_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Phase 2: Bounding Boxes", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Phase 2 Complete.")

if __name__ == "__main__":
    phase2_visuals()
