import cv2
import mediapipe as mp
import csv

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5)

# Start the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set video capture properties for resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Labels for each sign (customize this with the signs you want to recognize)
sign_labels = {0: 'A', 1: 'B', 2: 'C'}

# Open CSV file for saving data
with open('sign_language_data.csv', 'w', newline='') as f:
    writer = csv.writer(f)

    # CSV headers: each landmark (x, y) for all 21 hand landmarks, plus the label
    headers = []
    for i in range(21):
        headers.append(f'x{i}')
        headers.append(f'y{i}')
    headers.append('label')
    writer.writerow(headers)

    frame_skip = 2  # Process every 2nd frame
    frame_count = 0
    current_label = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue  # Skip this frame

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Collect landmark data
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append(lm.x)
                    landmarks.append(lm.y)

                # Check for key presses for labels
                key = cv2.waitKey(1) & 0xFF
                if key == ord('0'):
                    current_label = 0
                elif key == ord('1'):
                    current_label = 1
                elif key == ord('2'):
                    current_label = 2

                if current_label is not None:
                    landmarks.append(current_label)
                    writer.writerow(landmarks)
                    print(f"Recorded sign: {sign_labels[current_label]}")
                    current_label = None  # Reset the label after recording

        # Show webcam feed with landmarks
        cv2.imshow('Hand Tracking', frame)

        # Allow quitting the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
