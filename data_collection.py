import cv2
import mediapipe as mp
import csv

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

sign_labels = {i: chr(65 + i) for i in range(26)}  # 65 is ASCII for 'A'

try:
    with open('sign_language_data.csv', 'w', newline='') as f:
        writer = csv.writer(f)

        headers = []
        for i in range(21):
            headers.append(f'x{i}')
            headers.append(f'y{i}')
        headers.append('label')
        writer.writerow(headers)

        frame_interval = 1  # Capture every frame
        frame_count = 0
        current_label = 0
        frames_recorded = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image.")
                break

            frame_count += 1

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(frame_rgb)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.append(lm.x)
                        landmarks.append(lm.y)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord(' '):  # Press space to capture frames for the current alphabet
                        frames_recorded += 1
                        landmarks.append(current_label)
                        writer.writerow(landmarks)
                        print(f"Recorded sign: {sign_labels[current_label]} - Frames: {frames_recorded}/100", end='\r')

                        if frames_recorded >= 100:
                            print(f"\nRecorded 100 frames for sign: {sign_labels[current_label]}")
                            current_label = (current_label + 1) % 26
                            frames_recorded = 0

                            if current_label == 0:
                                break  # Stop after collecting frames for all alphabets

            cv2.imshow('Hand Tracking', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

except csv.Error as e:
    print(f"Error writing to CSV file: {e}")

finally:
    cap.release()
    cv2.destroyAllWindows()