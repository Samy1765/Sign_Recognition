import cv2
import mediapipe as mp
import pickle
import numpy as np

# Load the trained model
with open('sign_language_model.p', 'rb') as f:
    model_dict = pickle.load(f)
model = model_dict['model']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

# Define the labels where 65 is the ascii value of 'A'.
labels_dict = {i: chr(65 + i)for i in range(26)}

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmarks as features
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)

            # Convert to numpy array and reshape for prediction
            data_aux_np = np.asarray(landmarks).reshape(1, -1)

            # Predict the sign
            prediction = model.predict(data_aux_np)

            # Get the predicted label
            predicted_label = labels_dict[prediction[0]]
            print(f"Predicted Sign: {predicted_label}")

            # Display the predicted label on the frame
            cv2.putText(frame, predicted_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Sign Language Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
