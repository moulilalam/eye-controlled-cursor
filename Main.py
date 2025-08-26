import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# Initialize video capture and Face Mesh with refined landmarks.chr
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()

# Flags and variables for blink detection
left_blink_detected = False
right_blink_detected = False

# Double blink detection for left eye
double_left_blink_counter = 0
last_left_blink_time = 0
DOUBLE_BLINK_TIMEOUT = 0.7  # Seconds to detect double blink
# Blink threshold (calibrate as needed)
blink_threshold = 0.015
while True:
    ret, frame = cam.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    frame_h, frame_w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    current_time = time.time()

    if landmark_points:
        landmarks = landmark_points[0].landmark

        # Cursor Movement using right iris (indices 474-477)
        iris_points = landmarks[474:478]
        iris_center_x = int(np.mean([p.x for p in iris_points]) * frame_w)
        iris_center_y = int(np.mean([p.y for p in iris_points]) * frame_h)
        cv2.circle(frame, (iris_center_x, iris_center_y), 3, (0, 255, 0), -1)
        screen_x = int(screen_w * iris_center_x / frame_w)
        screen_y = int(screen_h * iris_center_y / frame_h)
        pyautogui.moveTo(screen_x, screen_y)

        # Blink Detection
        left_eye_distance = abs(landmarks[145].y - landmarks[159].y)
        right_eye_distance = abs(landmarks[374].y - landmarks[386].y)

        # Left Eye Blink (Single Click & Double Click)
        if left_eye_distance < blink_threshold and not left_blink_detected:
            left_blink_detected = True
            if double_left_blink_counter == 0:
                double_left_blink_counter = 1
                last_left_blink_time = current_time
            elif double_left_blink_counter == 1:
                if current_time - last_left_blink_time < DOUBLE_BLINK_TIMEOUT:
                    print("Double left blink detected, executing double click")
                    pyautogui.doubleClick(button='left')
                    double_left_blink_counter = 0
                else:
                    double_left_blink_counter = 1
                    last_left_blink_time = current_time
        elif left_eye_distance >= blink_threshold:
            left_blink_detected = False

        if double_left_blink_counter == 1 and (current_time - last_left_blink_time) > DOUBLE_BLINK_TIMEOUT:
            print("Left single blink detected, executing left click")
            pyautogui.click(button='left')
            double_left_blink_counter = 0

        # Right Eye Blink (Right Click)
        if right_eye_distance < blink_threshold and not right_blink_detected:
            print("Right blink detected, executing right click")
            pyautogui.click(button='right')
            right_blink_detected = True
        elif right_eye_distance >= blink_threshold:
            right_blink_detected = False

        # Visualization of Blink Landmarks
        left_points = [(int(landmarks[145].x * frame_w), int(landmarks[145].y * frame_h)),
                       (int(landmarks[159].x * frame_w), int(landmarks[159].y * frame_h))]
        right_points = [(int(landmarks[374].x * frame_w), int(landmarks[374].y * frame_h)),
                        (int(landmarks[386].x * frame_w), int(landmarks[386].y * frame_h))]
        for point in left_points:
            cv2.circle(frame, point, 3, (255, 0, 0), -1)
        for point in right_points:
            cv2.circle(frame, point, 3, (0, 0, 255), -1)

    cv2.imshow('Eye Control Cursor', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()