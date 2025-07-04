from ultralytics import YOLO
import cv2
import numpy as np
import time

# Load YOLOv8 model
model = YOLO("yolov8s.pt")

# Open webcam
cap = cv2.VideoCapture(0)

BALL_CLASS_ID = 37
trajectory_points = []  # stores (x, y, timestamp)
FADE_TIME = 2.0  # seconds

# Virtual "rim" settings (right side of screen)
shot_zone_center = (580, 240)  # adjust depending on resolution (Right center)
shot_zone_radius = 40
shots_made = 0
already_counted = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()

    # 🪞 Mirror the frame horizontally
    frame = cv2.flip(frame, 1)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    ball_position = None
    ball_detected = False

    # --- YOLOv8 Detection ---
    results = model.predict(frame, imgsz=640, conf=0.5)[0]
    for box in results.boxes:
        cls_id = int(box.cls[0].item())
        if cls_id == BALL_CLASS_ID:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            ball_position = (cx, cy)
            ball_detected = True

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, ball_position, 5, (0, 0, 255), -1)
            cv2.putText(frame, "AI: Basketball", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            break

    # --- Yellow Color Detection (Fallback) ---
    if not ball_detected:
        for cnt in yellow_contours:
            area = cv2.contourArea(cnt)
            if area > 500:
                x, y, w, h = cv2.boundingRect(cnt)
                cx = x + w // 2
                cy = y + h // 2
                ball_position = (cx, cy)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.circle(frame, ball_position, 5, (0, 0, 255), -1)
                cv2.putText(frame, "Color: Yellow Ball", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                break

    # --- Trajectory + Shot Counter ---
    if ball_position:
        trajectory_points.append((ball_position[0], ball_position[1], current_time))

        # Check for made shot
        dist = np.linalg.norm(np.array(ball_position) - np.array(shot_zone_center))
        if dist < shot_zone_radius:
            if not already_counted:
                shots_made += 1
                already_counted = True
        else:
            already_counted = False

    # Remove old trail points
    trajectory_points = [(x, y, t) for x, y, t in trajectory_points if current_time - t <= FADE_TIME]

    # Draw fading trajectory
    for x, y, t in trajectory_points:
        fade = 1 - (current_time - t) / FADE_TIME
        color = (int(255 * fade), 0, int(255 * (1 - fade)))
        cv2.circle(frame, (x, y), 3, color, -1)

    # Draw virtual rim
    cv2.circle(frame, shot_zone_center, shot_zone_radius, (255, 255, 255), 2)
    cv2.putText(frame, f"Shots Made: {shots_made}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    cv2.imshow("Basketball Tracker", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
