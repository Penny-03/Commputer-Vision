import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize drawing canvas
canvas = None

# Drawing parameters
draw_color = (0, 255, 0)  # Green color for drawing
brush_thickness = 5
eraser_thickness = 50

# Previous position for smooth drawing
prev_x, prev_y = None, None

# Drawing mode flag
drawing_mode = True  # True for draw, False for erase

def get_finger_tip_position(hand_landmarks, frame_shape):
    """Get the index finger tip position in pixel coordinates"""
    # Index finger tip is landmark 8
    index_tip = hand_landmarks.landmark[8]
    h, w, _ = frame_shape
    x = int(index_tip.x * w)
    y = int(index_tip.y * h)
    return x, y

def is_finger_up(hand_landmarks, finger_tip_id, finger_pip_id):
    """Check if a finger is extended (pointing up)"""
    tip = hand_landmarks.landmark[finger_tip_id]
    pip = hand_landmarks.landmark[finger_pip_id]
    return tip.y < pip.y  # Tip is above PIP joint

def get_gesture(hand_landmarks):
    """Detect basic gestures for drawing control"""
    # Index finger: landmark 8 (tip), 6 (PIP)
    # Middle finger: landmark 12 (tip), 10 (PIP)
    # Thumb: landmark 4 (tip), 3 (IP)
    
    index_up = is_finger_up(hand_landmarks, 8, 6)
    middle_up = is_finger_up(hand_landmarks, 12, 10)
    ring_up = is_finger_up(hand_landmarks, 16, 14)
    pinky_up = is_finger_up(hand_landmarks, 20, 18)
    
    # Only index finger up = draw
    if index_up and not middle_up and not ring_up and not pinky_up:
        return "draw"
    # Index and middle fingers up = erase
    elif index_up and middle_up and not ring_up and not pinky_up:
        return "erase"
    # All fingers up = clear canvas
    elif index_up and middle_up and ring_up and pinky_up:
        return "clear"
    else:
        return "none"

# Start video capture
cap = cv2.VideoCapture(0)

# Get frame dimensions for canvas
ret, frame = cap.read()
if ret:
    canvas = np.zeros_like(frame)

# Initialize MediaPipe Hands
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=1) as hands:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe
        results = hands.process(rgb_frame)
        
        # Draw hand landmarks and process gestures
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks on frame
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                
                # Get gesture
                gesture = get_gesture(hand_landmarks)
                
                # Get index finger tip position
                x, y = get_finger_tip_position(hand_landmarks, frame.shape)
                
                # Draw a circle at finger tip
                if gesture == "draw":
                    cv2.circle(frame, (x, y), 10, draw_color, -1)
                    
                    # Draw on canvas
                    if prev_x is not None and prev_y is not None:
                        cv2.line(canvas, (prev_x, prev_y), (x, y), 
                                draw_color, brush_thickness)
                    prev_x, prev_y = x, y
                    
                elif gesture == "erase":
                    cv2.circle(frame, (x, y), eraser_thickness//2, (0, 0, 255), 2)
                    
                    # Erase on canvas
                    if prev_x is not None and prev_y is not None:
                        cv2.line(canvas, (prev_x, prev_y), (x, y), 
                                (0, 0, 0), eraser_thickness)
                    prev_x, prev_y = x, y
                    
                elif gesture == "clear":
                    canvas = np.zeros_like(frame)
                    cv2.putText(frame, "CLEARING!", (50, 100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                    prev_x, prev_y = None, None
                    
                else:
                    prev_x, prev_y = None, None
        else:
            prev_x, prev_y = None, None
        
        # Combine frame and canvas
        combined = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)
        
        # Add instructions
        cv2.putText(combined, "Index finger: Draw", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined, "Index + Middle: Erase", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined, "All fingers up: Clear", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined, "Press 'q' to quit", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display the result
        cv2.imshow('Hand Drawing', combined)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Cleanup
cap.release()
cv2.destroyAllWindows()