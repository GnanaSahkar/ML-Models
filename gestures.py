import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Initialize Video Capture
cap = cv2.VideoCapture(0)

def finger_status(hand_landmarks):
    """
    Determine which fingers are up.
    """
    # Landmarks indices for finger tips
    tip_ids = [4, 8, 12, 16, 20]
    
    fingers = []
    # Thumb
    if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other fingers
    for id in range(1, 5):
        if hand_landmarks.landmark[tip_ids[id]].y < hand_landmarks.landmark[tip_ids[id] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    
    return fingers

def get_gesture_text(fingers):
    """
    Get text based on which fingers are up.
    """
    combinations = {
        (0, 0, 0, 0, 0): "No fingers",
        (1, 0, 0, 0, 0): "Thumb",
        (0, 1, 0, 0, 0): "Index",
        (0, 0, 1, 0, 0): "Middle",
        (0, 0, 0, 1, 0): "Ring",
        (0, 0, 0, 0, 1): "Pinky",
        (1, 1, 0, 0, 0): "Thumb & Index",
        (1, 0, 1, 0, 0): "Thumb & Middle",
        (1, 0, 0, 1, 0): "Thumb & Ring",
        (1, 0, 0, 0, 1): "Thumb & Pinky",
        (0, 1, 1, 0, 0): "Index & Middle",
        (0, 1, 0, 1, 0): "Index & Ring",
        (0, 1, 0, 0, 1): "Index & Pinky",
        (0, 0, 1, 1, 0): "Middle & Ring",
        (0, 0, 1, 0, 1): "Middle & Pinky",
        (0, 0, 0, 1, 1): "Ring & Pinky",
        (1, 1, 1, 0, 0): "Thumb, Index & Middle",
        (1, 1, 0, 1, 0): "Thumb, Index & Ring",
        (1, 1, 0, 0, 1): "Thumb, Index & Pinky",
        (1, 0, 1, 1, 0): "Thumb, Middle & Ring",
        (1, 0, 1, 0, 1): "Thumb, Middle & Pinky",
        (1, 0, 0, 1, 1): "Thumb, Ring & Pinky",
        (0, 1, 1, 1, 0): "Index, Middle & Ring",
        (0, 1, 1, 0, 1): "Index, Middle & Pinky",
        (0, 1, 0, 1, 1): "Index, Ring & Pinky",
        (0, 0, 1, 1, 1): "Middle, Ring & Pinky",
        (1, 1, 1, 1, 0): "Thumb, Index, Middle & Ring",
        (1, 1, 1, 0, 1): "Thumb, Index, Middle & Pinky",
        (1, 1, 0, 1, 1): "Thumb, Index, Ring & Pinky",
        (1, 0, 1, 1, 1): "Thumb, Middle, Ring & Pinky",
        (0, 1, 1, 1, 1): "Index, Middle, Ring & Pinky",
        (1, 1, 1, 1, 1): "All fingers"
    }
    
    return combinations.get(tuple(fingers), "Unknown gesture")

# Capture and Process Each Frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            fingers = finger_status(hand_landmarks)
            gesture_text = get_gesture_text(fingers)
            cv2.putText(frame, gesture_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    # Display the Frame
    cv2.imshow('Hand Gesture Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release Resources
cap.release()
cv2.destroyAllWindows()
