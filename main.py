import cv2
import time
from datetime import datetime


def initialize_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not detected.")
        exit()
    return cap


def calculate_fps(prev_frame_time):
    curr = time.time()
    fps = 1 / (curr - prev_frame_time) if prev_frame_time != 0 else 0
    return fps, curr


def draw_timestamp(frame):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, now, (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2)




def draw_zone_line(frame):
    h, w, _ = frame.shape
    cv2.line(frame, (0, h // 2), (w, h // 2), (0, 0, 255), 2)

def main():
    cap = initialize_camera()
    prev_frame_time = 0

    # ---- START IN NORMAL MODE ----
    window_name = "AI Vision - Phase 1"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1024, 576)

    # Aspect ratio lock
    TARGET_W = 1024
    TARGET_H = 576
    ASPECT = TARGET_W / TARGET_H

    fullscreen = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame.")
            break

        # --- Maintain Aspect Ratio ---
        h, w, _ = frame.shape
        current_aspect = w / h

        if abs(current_aspect - ASPECT) > 0.01:
            # scale to target aspect
            frame = cv2.resize(frame, (TARGET_W, TARGET_H))

        # --- FPS ---
        fps, prev_frame_time = calculate_fps(prev_frame_time)
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 2)

        # --- Timestamp ---
        draw_timestamp(frame)

        # --- Zone Line ---
        draw_zone_line(frame)

        # --- Display ---
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF

        # Quit
        if key == ord('q'):
            break

        # Mode 3 — Fullscreen toggle ('f')
        if key == ord('f'):
            cv2.setWindowProperty(window_name,
                                  cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN)
            fullscreen = True

        # Mode 1 — Normal window ('n')
        if key == ord('n'):
            cv2.setWindowProperty(window_name,
                                  cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 1024, 576)
            fullscreen = False

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()




