# hand_gesture_heuristic.py

import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Function to classify gestures based on landmark positions
def classify_gesture(landmarks):
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    wrist = landmarks[0]

    # Calculate distances from wrist to fingertips
    index_dist = index_tip.y - wrist.y
    middle_dist = middle_tip.y - wrist.y
    ring_dist = ring_tip.y - wrist.y
    pinky_dist = pinky_tip.y - wrist.y
    thumb_dist_x = thumb_tip.x - wrist.x

    # Simple heuristic rules
    if all(finger.y > wrist.y for finger in [index_tip, middle_tip, ring_tip, pinky_tip]) and thumb_dist_x < 0.05:
        return "Fist"
    elif all(finger.y < wrist.y for finger in [index_tip, middle_tip, ring_tip, pinky_tip]):
        return "HI"
    elif thumb_tip.y < wrist.y and index_tip.y > wrist.y:
        return "Thumbs Up"
    elif index_tip.y < wrist.y and middle_tip.y < wrist.y and ring_tip.y > wrist.y and pinky_tip.y > wrist.y:
        return "Victory"
    else:
        return "Unknown"

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Convert landmarks to a list of (x,y,z)
            lm = hand_landmarks.landmark
            gesture = classify_gesture(lm)

            cv2.putText(frame, gesture, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Hand Gesture Recognition (Heuristic)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()