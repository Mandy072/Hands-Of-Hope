import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import yaml

# Function to preprocess hand landmarks
def preprocess_hand_landmarks(hand_landmarks):
    landmarks = [point['x'] for point in hand_landmarks['hand_landmarks']]
    landmarks += [point['y'] for point in hand_landmarks['hand_landmarks']]
    landmarks += [point['z'] for point in hand_landmarks['hand_landmarks']]
    return np.array(landmarks)

# Load the trained model
model = tf.keras.models.load_model('Dataset/sign_language_model.h5')

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

cap = cv2.VideoCapture(0)

# Set the margin for the bounding box
margin = 20

while True:
    _, frame = cap.read()

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image with Mediapipe Hands
    results = hands.process(rgb_frame)

    # Check if hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract bounding box coordinates with margin
            bbox_min = [float('inf'), float('inf')]  # Initialize with large values
            bbox_max = [float('-inf'), float('-inf')]  # Initialize with small values

            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])

                # Update bounding box coordinates
                bbox_min[0] = min(bbox_min[0], x)
                bbox_min[1] = min(bbox_min[1], y)
                bbox_max[0] = max(bbox_max[0], x)
                bbox_max[1] = max(bbox_max[1], y)

            # Add margin to the bounding box
            bbox_min[0] = max(0, bbox_min[0] - margin)
            bbox_min[1] = max(0, bbox_min[1] - margin)
            bbox_max[0] = min(frame.shape[1], bbox_max[0] + margin)
            bbox_max[1] = min(frame.shape[0], bbox_max[1] + margin)

            # Draw bounding box
            cv2.rectangle(frame, (int(bbox_min[0]), int(bbox_min[1])),
                          (int(bbox_max[0]), int(bbox_max[1])), (0, 255, 0), 2)

            # Capture hand landmarks
            hand_landmarks_data = {'hand_landmarks': []}
            for landmark in hand_landmarks.landmark:
                hand_landmarks_data['hand_landmarks'].append({'x': landmark.x, 'y': landmark.y, 'z': landmark.z})

            # Preprocess hand landmarks
            # hand_landmarks_processed = preprocess_hand_landmarks(hand_landmarks_data)

            # Assuming hand_landmarks_processed has shape (None, 21)
            hand_landmarks_processed = preprocess_hand_landmarks(hand_landmarks_data)

            # Reshape to match the expected input shape (None, 21)
            hand_landmarks_processed = np.reshape(hand_landmarks_processed, (1, -1))
                        # Make prediction
            prediction = model.predict(hand_landmarks_processed)
            predicted_class = np.argmax(prediction)

            # Display the predicted sign
            cv2.putText(frame, f"Predicted Sign: {predicted_class}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the result
    cv2.imshow("Sign Prediction", frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
