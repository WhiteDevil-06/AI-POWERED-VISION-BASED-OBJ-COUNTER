import cv2
import time
import logging
from ultralytics import YOLO

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("Phase1")

def phase1_inference():
    print("--- PHASE 1: MODEL LOADING & BASIC DETECTION ---")
    
    # 1. Load Model
    print("Step 1: Loading YOLOv8n model (this may take a moment)...")
    try:
        model = YOLO("yolov8n.pt")
        print("SUCCESS: Model loaded.")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load model. {e}")
        return

    # 2. Initialize Camera
    cap = cv2.VideoCapture(0)
    width, height = 640, 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        print("Error: Camera not accessible.")
        return

    print("Step 2: Starting Inference Loop (Console Output Only).")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Frame is read, let's run inference
        # Conf threshold 0.55 as requested
        results = model(frame, verbose=False, conf=0.55)
        
        # Parse results for console output
        # We only care about box-like classes for the Logistcis Project, 
        # but for this phase, let's print EVERYTHING to ensure model works.
        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                conf = float(box.conf[0])
                detections.append(f"{label} ({conf:.2f})")
        
        if detections:
            print(f"Detected: {', '.join(detections)}")
        else:
            print("No detections...")

        # Optional: Show video just so user knows it's running (NO BOXES DRAWN)
        cv2.putText(frame, "PHASE 1: CONSOLE LOG ONLY", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Phase 1: Inference Check", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Phase 1 Complete.")

if __name__ == "__main__":
    phase1_inference()
