import cv2
import numpy as np
import mediapipe as mp
import csv
from model import KeyPointClassifier
from collections import deque
import time

# Load the KeyPointClassifier
try:
    keypoint_classifier = KeyPointClassifier()
except Exception as e:
    print(f"Error loading KeyPointClassifier: {e}")
    exit(1)

# Set up MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
try:
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
except Exception as e:
    print(f"Error initializing MediaPipe Hands: {e}")
    exit(1)

# Load gesture labels from CSV
gesture_labels = []
csv_path = 'model/keypoint_classifier/keypoint_classifier_label.csv'
try:
    with open(csv_path, encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        for row in reader:
            gesture_labels.append(row[0])
except FileNotFoundError:
    print(f"Error: CSV file not found at {csv_path}")
    exit(1)
except Exception as e:
    print(f"Error reading CSV: {e}")
    exit(1)

# Start webcam
try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Could not open webcam")
except Exception as e:
    print(f"Error initializing webcam: {e}")
    exit(1)

def prepare_keypoints(landmarks, image_shape):
    """Convert MediaPipe landmarks to normalized keypoints."""
    image_width, image_height = image_shape[1], image_shape[0]
    landmark_list = []
    for landmark in landmarks.landmark:
        x = min(int(landmark.x * image_width), image_width - 1)
        y = min(int(landmark.y * image_height), image_height - 1)
        landmark_list.append([x, y])
    
    base_x, base_y = landmark_list[0]
    relative_list = [[point[0] - base_x, point[1] - base_y] for point in landmark_list]
    flat_list = [coord for point in relative_list for coord in point]
    max_value = max(abs(min(flat_list, default=0)), abs(max(flat_list, default=0)), 1)
    normalized_list = [x / max_value for x in flat_list]
    return normalized_list

def predict_gesture(keypoints):
    """Predict gesture ID using KeyPointClassifier."""
    try:
        gesture_id = keypoint_classifier(keypoints)
        return gesture_id
    except Exception as e:
        print(f"Error in gesture prediction: {e}")
        return None

def main():
    print("Starting ASL word recognition app...")
    print(f"Loaded {len(gesture_labels)} gesture labels")
    print("Press 'q' to quit")
    
    # Initialize variables
    letter_queue = deque(maxlen=5)  # Store up to 5 letters
    detection_count = 0
    required_detections = 10  # Frames needed to confirm a letter
    no_detection_timeout = 3.0  # Seconds before resetting word
    last_detection_time = time.time()
    last_gesture_id = None
    current_word = ""
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Could not read from camera")
            break
            
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        current_time = time.time()
        
        if results.multi_hand_landmarks:
            last_detection_time = current_time
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Prepare and predict gesture
                input_data = prepare_keypoints(hand_landmarks, frame.shape)
                gesture_id = predict_gesture(input_data)
                
                if gesture_id is None:
                    continue
                
                # Require consistent detections
                if gesture_id == last_gesture_id:
                    detection_count += 1
                else:
                    detection_count = 1
                    last_gesture_id = gesture_id
                
                if detection_count >= required_detections:
                    letter = gesture_labels[gesture_id]
                    if len(letter_queue) == 0 or letter_queue[-1] != letter:
                        letter_queue.append(letter)
                        current_word = "".join(letter_queue).lower()
                    detection_count = 0
                
                # Display current letter and word
                cv2.putText(frame, f"Current: {gesture_labels[gesture_id]}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Word: {current_word}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        else:
            detection_count = 0
            last_gesture_id = None
            # Reset word after timeout
            if current_time - last_detection_time >= no_detection_timeout:
                letter_queue.clear()
                current_word = ""
        
        cv2.imshow('ASL Word Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        cap.release()
        cv2.destroyAllWindows()