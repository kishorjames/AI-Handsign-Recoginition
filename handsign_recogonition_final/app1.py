import cv2
import numpy as np
import mediapipe as mp
import csv
from model import KeyPointClassifier  # Assuming this is in a file called model.py

# Load the KeyPointClassifier (assumes it handles the .tflite model internally)
keypoint_classifier = KeyPointClassifier()

# Set up MediaPipe Hands for detecting hand keypoints
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils  # For drawing hand landmarks

# Load gesture labels from the CSV file
gesture_labels = []
csv_path = 'model/keypoint_classifier/keypoint_classifier_label.csv'
with open(csv_path, encoding='utf-8-sig') as f:
    reader = csv.reader(f)
    for row in reader:
        gesture_labels.append(row[0])  # Take the first column of each row

# Start the webcam
cap = cv2.VideoCapture(0)  # 0 is the default camera

# Function to prepare keypoints for the model (similar to pre_process_landmark in outsourced code)
def prepare_keypoints(landmarks):
    # Convert landmarks to a list of pixel coordinates
    image_width, image_height = frame.shape[1], frame.shape[0]
    landmark_list = []
    for landmark in landmarks.landmark:
        x = min(int(landmark.x * image_width), image_width - 1)
        y = min(int(landmark.y * image_height), image_height - 1)
        landmark_list.append([x, y])
    
    # Convert to relative coordinates (relative to the first keypoint)
    base_x, base_y = landmark_list[0]
    relative_list = []
    for point in landmark_list:
        relative_list.append([point[0] - base_x, point[1] - base_y])
    
    # Flatten the list (turn [[x1, y1], [x2, y2], ...] into [x1, y1, x2, y2, ...])
    flat_list = [coord for point in relative_list for coord in point]
    
    # Normalize the values (divide by the maximum absolute value)
    max_value = max(abs(min(flat_list)), abs(max(flat_list)), 1)  # Avoid division by zero
    normalized_list = [x / max_value for x in flat_list]
    
    return normalized_list

# Function to predict the gesture using KeyPointClassifier
def predict_gesture(keypoints):
    # Use the KeyPointClassifier to get the gesture ID
    gesture_id = keypoint_classifier(keypoints)
    # Note: Confidence isn't directly available with this method unless the class provides it
    return gesture_id, None  # Returning None for confidence since it’s not in the original

# Main loop to capture video and recognize gestures
def main():
    print("Starting the hand gesture recognition app...")
    print("Loaded", len(gesture_labels), "gesture labels from CSV")
    print("Press 'q' to quit")

    global frame  # Make frame global so prepare_keypoints can access image dimensions
    while True:
        # Capture a frame from the webcam
        ret, frame = cap.read()
        if not ret:  # If no frame is captured, stop
            print("Could not read from camera")
            break

        # Flip the frame so it looks like a mirror
        frame = cv2.flip(frame, 1)
        
        # Convert the frame to RGB (MediaPipe needs RGB, not BGR)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Use MediaPipe to find hand keypoints
        results = hands.process(rgb_frame)

        # If a hand is detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw the hand keypoints and connections on the frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Prepare the keypoints for the model
                input_data = prepare_keypoints(hand_landmarks)

                # Predict the gesture
                gesture_id, confidence = predict_gesture(input_data)

                # Display the predicted gesture on the frame
                if confidence is not None:
                    text = f"Gesture: {gesture_labels[gesture_id]} ({confidence:.2f})"
                else:
                    text = f"Gesture: {gesture_labels[gesture_id]}"
                cv2.putText(frame, text, (10, 30),  # Position at top-left
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the frame with the gesture prediction
        cv2.imshow('Hand Gesture Recognition', frame)

        # Quit the app if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up: release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

# Run the app
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Something went wrong: {e}")
        cap.release()
        cv2.destroyAllWindows()
