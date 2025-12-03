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
draw_color = (0, 255, 0)  # Default green color
brush_thickness = 5
eraser_thickness = 50

# Previous position for smooth drawing
prev_x, prev_y = None, None

# UI Button class
class Button:
    def __init__(self, x, y, w, h, color, text="", action=None):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.color = color
        self.text = text
        self.action = action
        self.active = False
        
    def draw(self, frame):
        # Draw button background
        thickness = -1 if self.active else -1
        cv2.rectangle(frame, (self.x, self.y), (self.x + self.w, self.y + self.h), 
                     self.color, thickness)
        
        # Draw border (thicker if active)
        border_thickness = 4 if self.active else 2
        cv2.rectangle(frame, (self.x, self.y), (self.x + self.w, self.y + self.h), 
                     (255, 255, 255), border_thickness)
        
        # Draw text if any
        if self.text:
            font_scale = 0.5
            font_thickness = 2
            text_size = cv2.getTextSize(self.text, cv2.FONT_HERSHEY_SIMPLEX, 
                                       font_scale, font_thickness)[0]
            text_x = self.x + (self.w - text_size[0]) // 2
            text_y = self.y + (self.h + text_size[1]) // 2
            cv2.putText(frame, self.text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 
                       font_thickness)
    
    def is_clicked(self, x, y):
        return self.x <= x <= self.x + self.w and self.y <= y <= self.y + self.h

class Slider:
    def __init__(self, x, y, w, h, min_val, max_val, current_val, label=""):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.min_val = min_val
        self.max_val = max_val
        self.current_val = current_val
        self.label = label
        self.dragging = False
        
    def draw(self, frame):
        # Draw slider background
        cv2.rectangle(frame, (self.x, self.y), (self.x + self.w, self.y + self.h), 
                     (100, 100, 100), -1)
        
        # Calculate knob position
        value_ratio = (self.current_val - self.min_val) / (self.max_val - self.min_val)
        knob_x = int(self.x + value_ratio * self.w)
        
        # Draw filled portion
        cv2.rectangle(frame, (self.x, self.y), (knob_x, self.y + self.h), 
                     (0, 200, 0), -1)
        
        # Draw knob
        cv2.circle(frame, (knob_x, self.y + self.h // 2), 12, (255, 255, 255), -1)
        cv2.circle(frame, (knob_x, self.y + self.h // 2), 12, (0, 0, 0), 2)
        
        # Draw label and value
        if self.label:
            cv2.putText(frame, f"{self.label}: {int(self.current_val)}", 
                       (self.x, self.y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 255, 255), 2)
    
    def update(self, x):
        if self.x <= x <= self.x + self.w:
            value_ratio = (x - self.x) / self.w
            self.current_val = self.min_val + value_ratio * (self.max_val - self.min_val)
            return True
        return False
    
    def is_clicked(self, x, y):
        knob_x = int(self.x + ((self.current_val - self.min_val) / 
                              (self.max_val - self.min_val)) * self.w)
        distance = np.sqrt((x - knob_x)**2 + (y - self.y - self.h//2)**2)
        return distance <= 15 or (self.x <= x <= self.x + self.w and 
                                 self.y <= y <= self.y + self.h)

# Create UI elements
color_buttons = [
    Button(10, 150, 50, 50, (0, 0, 255), "Red"),      # Red
    Button(70, 150, 50, 50, (0, 255, 0), "Green"),    # Green
    Button(130, 150, 50, 50, (255, 0, 0), "Blue"),    # Blue
    Button(190, 150, 50, 50, (0, 255, 255), "Yellow"), # Yellow
    Button(250, 150, 50, 50, (255, 0, 255), "Pink"),  # Magenta
    Button(310, 150, 50, 50, (255, 255, 255), "White"), # White
    Button(370, 150, 50, 50, (0, 0, 0), "Black"),     # Black
]

# Tool buttons
clear_button = Button(10, 220, 100, 50, (200, 200, 200), "Clear")
eraser_button = Button(120, 220, 100, 50, (150, 150, 150), "Eraser")

# Sliders
brush_slider = Slider(10, 300, 200, 20, 1, 30, brush_thickness, "Brush")
eraser_slider = Slider(10, 350, 200, 20, 10, 100, eraser_thickness, "Eraser")

# Current tool
current_tool = "draw"  # "draw" or "erase"

def get_finger_tip_position(hand_landmarks, frame_shape):
    """Get the index finger tip position in pixel coordinates"""
    index_tip = hand_landmarks.landmark[8]
    h, w, _ = frame_shape
    x = int(index_tip.x * w)
    y = int(index_tip.y * h)
    return x, y

def is_finger_up(hand_landmarks, finger_tip_id, finger_pip_id):
    """Check if a finger is extended"""
    tip = hand_landmarks.landmark[finger_tip_id]
    pip = hand_landmarks.landmark[finger_pip_id]
    return tip.y < pip.y

def get_gesture(hand_landmarks):
    """Detect gestures"""
    index_up = is_finger_up(hand_landmarks, 8, 6)
    middle_up = is_finger_up(hand_landmarks, 12, 10)
    
    # Only index finger up = interact/draw
    if index_up and not middle_up:
        return "draw"
    else:
        return "none"

def check_ui_interaction(x, y):
    """Check if finger is clicking on any UI element"""
    global draw_color, brush_thickness, eraser_thickness, canvas, current_tool
    
    # Check color buttons
    for btn in color_buttons:
        if btn.is_clicked(x, y):
            draw_color = btn.color
            current_tool = "draw"
            # Reset all button active states
            for b in color_buttons:
                b.active = False
            btn.active = True
            return True
    
    # Check clear button
    if clear_button.is_clicked(x, y):
        canvas = np.zeros_like(canvas)
        return True
    
    # Check eraser button
    if eraser_button.is_clicked(x, y):
        current_tool = "erase"
        eraser_button.active = True
        return True
    else:
        eraser_button.active = False
    
    # Check brush slider
    if brush_slider.is_clicked(x, y):
        brush_slider.dragging = True
        brush_slider.update(x)
        brush_thickness = int(brush_slider.current_val)
        return True
    
    # Check eraser slider
    if eraser_slider.is_clicked(x, y):
        eraser_slider.dragging = True
        eraser_slider.update(x)
        eraser_thickness = int(eraser_slider.current_val)
        return True
    
    return False

def draw_ui(frame):
    """Draw all UI elements"""
    # Draw semi-transparent panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 140), (450, 400), (50, 50, 50), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Draw title
    cv2.putText(frame, "Tools", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 
               0.7, (255, 255, 255), 2)
    
    # Draw all buttons
    for btn in color_buttons:
        btn.draw(frame)
    
    clear_button.draw(frame)
    eraser_button.draw(frame)
    
    # Draw sliders
    brush_slider.draw(frame)
    eraser_slider.draw(frame)
    
    # Draw current color indicator
    cv2.putText(frame, "Current:", (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 
               0.6, (255, 255, 255), 2)
    cv2.rectangle(frame, (100, 420), (150, 450), draw_color, -1)
    cv2.rectangle(frame, (100, 420), (150, 450), (255, 255, 255), 2)
    
    # Draw instructions
    cv2.putText(frame, "Point with index finger to draw/click", (10, 480), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "Press 'q' to quit | 's' to save", (10, 500), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# Start video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Get frame dimensions for canvas
ret, frame = cap.read()
if ret:
    canvas = np.zeros_like(frame)

# Set first color button as active
color_buttons[1].active = True  # Green by default

# Click cooldown
click_cooldown = 0
click_delay = 15  # frames

# Initialize MediaPipe Hands
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
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
        
        # Decrease click cooldown
        if click_cooldown > 0:
            click_cooldown -= 1
        
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
                
                if gesture == "draw":
                    # Check UI interaction first
                    ui_clicked = check_ui_interaction(x, y)
                    
                    if ui_clicked and click_cooldown == 0:
                        click_cooldown = click_delay
                        prev_x, prev_y = None, None
                    elif not ui_clicked:
                        # Drawing on canvas
                        if current_tool == "draw":
                            cv2.circle(frame, (x, y), brush_thickness, draw_color, -1)
                            if prev_x is not None and prev_y is not None:
                                cv2.line(canvas, (prev_x, prev_y), (x, y), 
                                        draw_color, brush_thickness)
                        elif current_tool == "erase":
                            cv2.circle(frame, (x, y), eraser_thickness//2, (0, 0, 255), 2)
                            if prev_x is not None and prev_y is not None:
                                cv2.line(canvas, (prev_x, prev_y), (x, y), 
                                        (0, 0, 0), eraser_thickness)
                        
                        prev_x, prev_y = x, y
                    else:
                        # Show pointer on UI
                        cv2.circle(frame, (x, y), 10, (0, 255, 255), -1)
                else:
                    prev_x, prev_y = None, None
                    brush_slider.dragging = False
                    eraser_slider.dragging = False
        else:
            prev_x, prev_y = None, None
            brush_slider.dragging = False
            eraser_slider.dragging = False
        
        # Combine frame and canvas
        combined = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)
        
        # Draw UI on top
        draw_ui(combined)
        
        # Display the result
        cv2.imshow('Hand Drawing', combined)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save the drawing
            cv2.imwrite('drawing.png', canvas)
            print("Drawing saved as 'drawing.png'")

# Cleanup
cap.release()
cv2.destroyAllWindows()