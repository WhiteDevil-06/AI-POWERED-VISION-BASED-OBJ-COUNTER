import cv2
import time

def phase0_validation():
    print("--- PHASE 0: CAMERA VALIDATION ---")
    print("Initializing Camera (Source 0)...")
    
    cap = cv2.VideoCapture(0)
    
    # Check if opened
    if not cap.isOpened():
        print("CRITICAL ERROR: Could not open webcam.")
        return

    # Set Resolution (Best effort for 640x480 as per specs)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("SUCCESS: Camera initialized.")
    print("Press 'q' to quit.")

    prev_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame.")
            break

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time

        # Visuals (Text only, NO AI boxes)
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.putText(frame, "PHASE 0: VALIDATION", (10, 460), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Phase 0: Camera Check", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Phase 0 Complete.")

if __name__ == "__main__":
    phase0_validation()
