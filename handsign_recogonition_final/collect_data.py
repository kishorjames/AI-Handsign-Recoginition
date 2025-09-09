import csv
import copy
import argparse
import cv2 as cv
import numpy as np
import mediapipe as mp
from collections import deque

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--min_detection_confidence", type=float, default=0.7)
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5)
    parser.add_argument("--output_csv", type=str, default="keypoint_data.csv")
    parser.add_argument("--max_frames", type=int, default=1000, help="Max frames to collect per gesture")
    args = parser.parse_args()
    return args

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0, 2), int)
    for landmark in landmarks.landmark:
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point = [np.array((landmark_x, landmark_y))]
        landmark_array = np.append(landmark_array, landmark_point, axis=0)
    x, y, w, h = cv.boundingRect(landmark_array)
    return [x, y, x + w, y + h]

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for landmark in landmarks.landmark:
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = temp_landmark_list[0]
    for index, landmark_point in enumerate(temp_landmark_list):
        temp_landmark_list[index][0] -= base_x
        temp_landmark_list[index][1] -= base_y
    temp_landmark_list = [coord for point in temp_landmark_list for coord in point]
    max_value = max(list(map(abs, temp_landmark_list)), default=1)
    temp_landmark_list = [n / max_value for n in temp_landmark_list]
    return temp_landmark_list

def draw_bounding_rect(image, brect):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 255, 0), 1)
    return image

def draw_landmarks(image, landmark_point):
    for i, point in enumerate(landmark_point):
        cv.circle(image, (point[0], point[1]), 5, (0, 255, 0), -1)
    return image

def main():
    args = get_args()
    
    # Initialize camera
    cap = cv.VideoCapture(args.device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)
    
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )
    
    # CSV setup
    csv_header = ['gesture_id'] + [f'landmark_{i}_{xy}' for i in range(21) for xy in ['x', 'y']]
    gesture_counts = {}
    current_gesture = -1
    
    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
    
    print("Instructions:")
    print("Press 0-9 or a-z to select gesture ID (0-35)")
    print("Press 'q' to quit")
    print("Hold gesture until max frames collected or change gesture")
    
    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break
            
        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        
        # Process hand landmarks
        results = hands.process(image_rgb)
        
        # Handle keypress
        key = cv.waitKey(1)
        if key == ord('q'):
            break
        elif 48 <= key <= 57:  # 0-9
            current_gesture = key - 48
        elif 97 <= key <= 122:  # a-z
            current_gesture = key - 87
        
        if results.multi_hand_landmarks and current_gesture != -1:
            for hand_landmarks in results.multi_hand_landmarks:
                # Check frame limit for current gesture
                gesture_counts.setdefault(current_gesture, 0)
                if gesture_counts[current_gesture] >= args.max_frames:
                    continue
                
                # Calculate landmarks and bounding box
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                processed_landmarks = pre_process_landmark(landmark_list)
                
                # Write to CSV
                with open(args.output_csv, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([current_gesture] + processed_landmarks)
                
                gesture_counts[current_gesture] += 1
                
                # Draw visualizations
                debug_image = draw_bounding_rect(debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                
                # Display info
                cv.putText(debug_image, f"Gesture ID: {current_gesture}", (10, 30),
                          cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv.putText(debug_image, f"Frames: {gesture_counts[current_gesture]}/{args.max_frames}",
                          (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv.imshow('Keypoint Data Collector', debug_image)
    
    cap.release()
    cv.destroyAllWindows()
    hands.close()
    print(f"Data saved to {args.output_csv}")
    print("Gesture counts:", gesture_counts)

if __name__ == '__main__':
    main()