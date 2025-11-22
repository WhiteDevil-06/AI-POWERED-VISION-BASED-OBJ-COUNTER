import cv2
import time
from datetime import datetime

# -----------------------------
# PERSON 1 — CAMERA INITIALIZER
# -----------------------------
def initialize_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not detected.")
        exit()
    return cap

# -----------------------------
# PERSON 2 — FPS CALCULATOR
# -----------------------------
def calculate_fps(prev_frame_time):
    curr = time.time()
    fps = 1 / (curr - prev_frame_time) if prev_frame_time != 0 else 0
    return fps, curr

# -----------------------------
# PERSON 3 — TIMESTAMP OVERLAY
# -----------------------------
def draw_timestamp(frame):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, now, (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2)











# import cv2

# cap = cv2.VideoCapture(0)   # 0 = default webcam

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Camera feed not found!")
#         break

#     cv2.imshow("Webcam Test", frame)

#     # press 'q' to quit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
