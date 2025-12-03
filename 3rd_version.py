import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize drawing canvas
canvas = None
background_mode = "transparent"  # "transparent", "white", "color_by_numbers"

# Drawing parameters
draw_color = (0, 255, 0)  # Default green color
brush_thickness = 8
eraser_thickness = 50

# Previous position for smooth drawing
prev_x, prev_y = None, None

# UI state
show_color_palette = False
click_cooldown = 0
click_delay = 10

# UI Button class with modern styling
class Button:
    def __init__(self, x, y, w, h, color, text="", icon=None, action=None):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.color = color
        self.text = text
        self.icon = icon
        self.action = action
        self.active = False
        self.hover = False
        
    def draw(self, frame):
        # Create rounded rectangle effect
        overlay = frame.copy()
        
        # Determine colors based on state
        if self.active:
            bg_color = self.color
            border_color = (255, 255, 255)
            border_thickness = 3
        elif self.hover:
            # Lighten color for hover effect
            bg_color = tuple(min(255, int(c * 1.3)) for c in self.color)
            border_color = (200, 200, 200)
            border_thickness = 2
        else:
            bg_color = self.color
            border_color = (150, 150, 150)
            border_thickness = 1
        
        # Draw button background with rounded corners
        cv2.rectangle(overlay, (self.x + 2, self.y + 2), 
                     (self.x + self.w - 2, self.y + self.h - 2), 
                     bg_color, -1)
        cv2.rectangle(overlay, (self.x, self.y), 
                     (self.x + self.w, self.y + self.h), 
                     border_color, border_thickness)
        
        # Blend for semi-transparency if not active
        alpha = 0.95 if self.active else 0.85
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Draw text
        if self.text:
            font_scale = 0.5
            font_thickness = 2 if self.active else 1
            text_size = cv2.getTextSize(self.text, cv2.FONT_HERSHEY_SIMPLEX, 
                                       font_scale, font_thickness)[0]
            text_x = self.x + (self.w - text_size[0]) // 2
            text_y = self.y + (self.h + text_size[1]) // 2
            
            # Text shadow for better readability
            cv2.putText(frame, self.text, (text_x + 1, text_y + 1), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 
                       font_thickness)
            cv2.putText(frame, self.text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 
                       font_thickness)
    
    def is_over(self, x, y):
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
        self.hover = False
        
    def draw(self, frame):
        # Draw label
        if self.label:
            cv2.putText(frame, f"{self.label}", 
                       (self.x, self.y - 15), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"{int(self.current_val)}", 
                       (self.x + self.w + 10, self.y + self.h // 2 + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw slider track
        track_y = self.y + self.h // 2
        cv2.line(frame, (self.x, track_y), (self.x + self.w, track_y), 
                (80, 80, 80), self.h)
        
        # Calculate knob position
        value_ratio = (self.current_val - self.min_val) / (self.max_val - self.min_val)
        knob_x = int(self.x + value_ratio * self.w)
        
        # Draw filled track
        cv2.line(frame, (self.x, track_y), (knob_x, track_y), 
                (100, 200, 100), self.h)
        
        # Draw knob with glow effect if hovering
        if self.hover or self.dragging:
            cv2.circle(frame, (knob_x, track_y), 16, (150, 255, 150), -1)
        cv2.circle(frame, (knob_x, track_y), 12, (255, 255, 255), -1)
        cv2.circle(frame, (knob_x, track_y), 12, (100, 100, 100), 2)
    
    def update(self, x):
        if self.x <= x <= self.x + self.w:
            value_ratio = (x - self.x) / self.w
            self.current_val = self.min_val + value_ratio * (self.max_val - self.min_val)
            return True
        return False
    
    def is_over(self, x, y):
        track_y = self.y + self.h // 2
        knob_x = int(self.x + ((self.current_val - self.min_val) / 
                              (self.max_val - self.min_val)) * self.w)
        distance = np.sqrt((x - knob_x)**2 + (y - track_y)**2)
        return distance <= 20 or (self.x <= x <= self.x + self.w and 
                                 self.y - 10 <= y <= self.y + self.h + 10)

# Color palette for vertical sidebar
palette_colors = [
    ((255, 50, 50), "Red"),
    ((50, 255, 50), "Green"),
    ((50, 50, 255), "Blue"),
    ((255, 255, 50), "Yellow"),
    ((255, 50, 255), "Magenta"),
    ((50, 255, 255), "Cyan"),
    ((255, 128, 0), "Orange"),
    ((128, 0, 255), "Purple"),
    ((255, 255, 255), "White"),
    ((0, 0, 0), "Black"),
]

def create_color_by_numbers_template():
    """Create a simple color-by-numbers template"""
    template = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # Define regions with numbers
    regions = [
        # (x, y, radius, number, color)
        (400, 250, 80, 1, (255, 50, 50)),   # Red circle
        (600, 250, 80, 2, (50, 255, 50)),   # Green circle
        (800, 250, 80, 3, (50, 50, 255)),   # Blue circle
        (500, 450, 80, 4, (255, 255, 50)),  # Yellow circle
        (700, 450, 80, 5, (255, 50, 255)),  # Magenta circle
    ]
    
    # Draw outlines and numbers
    for x, y, r, num, color in regions:
        cv2.circle(template, (x, y), r, (200, 200, 200), 2)
        
        # Draw number in center
        font_scale = 1.5
        text = str(num)
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 3)[0]
        text_x = x - text_size[0] // 2
        text_y = y + text_size[1] // 2
        
        cv2.putText(template, text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (150, 150, 150), 3)
    
    # Add title and legend
    cv2.putText(template, "Color by Numbers", (50, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2)
    
    # Draw legend
    legend_y = 100
    for i, (x, y, r, num, color) in enumerate(regions):
        legend_x = 50
        legend_y_pos = legend_y + i * 40
        cv2.rectangle(template, (legend_x, legend_y_pos), 
                     (legend_x + 30, legend_y_pos + 30), color, -1)
        cv2.rectangle(template, (legend_x, legend_y_pos), 
                     (legend_x + 30, legend_y_pos + 30), (200, 200, 200), 2)
        cv2.putText(template, f"{num}", (legend_x + 50, legend_y_pos + 22), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    
    return template

def get_finger_tip_position(hand_landmarks, frame_shape):
    """Get the index finger tip position"""
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
    ring_up = is_finger_up(hand_landmarks, 16, 14)
    pinky_up = is_finger_up(hand_landmarks, 20, 18)
    
    # Only index finger = draw/interact
    if index_up and not middle_up and not ring_up and not pinky_up:
        return "draw"
    # Index and middle = erase
    elif index_up and middle_up and not ring_up and not pinky_up:
        return "erase"
    # All fingers = clear
    elif index_up and middle_up and ring_up and pinky_up:
        return "clear"
    else:
        return "none"

def draw_modern_panel(frame, x, y, w, h, alpha=0.85):
    """Draw a modern semi-transparent panel"""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (40, 40, 45), -1)
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (80, 80, 85), 2)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

def draw_color_palette(frame, x, y):
    """Draw vertical color palette"""
    panel_width = 140
    panel_height = len(palette_colors) * 55 + 70
    
    draw_modern_panel(frame, x, y, panel_width, panel_height, 0.92)
    
    # Title
    cv2.putText(frame, "Colors", (x + 15, y + 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Draw color swatches
    buttons = []
    for i, (color, name) in enumerate(palette_colors):
        btn_y = y + 50 + i * 55
        btn = Button(x + 10, btn_y, 120, 45, color, name)
        
        # Check if this is the current color
        if color == draw_color:
            btn.active = True
        
        btn.draw(frame)
        buttons.append(btn)
    
    return buttons

def draw_main_toolbar(frame, buttons, sliders):
    """Draw the main toolbar at the top"""
    toolbar_height = 100
    draw_modern_panel(frame, 0, 0, 1280, toolbar_height, 0.88)
    
    # Title
    cv2.putText(frame, "Advanced Hand Drawing Studio", (20, 35), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Draw all buttons
    for btn in buttons:
        btn.draw(frame)
    
    # Draw sliders
    for slider in sliders:
        slider.draw(frame)

# Start video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Get frame dimensions
ret, frame = cap.read()
if ret:
    canvas = np.zeros_like(frame)
    color_by_numbers_template = create_color_by_numbers_template()

# Create toolbar buttons
clear_btn = Button(20, 55, 90, 35, (220, 80, 80), "Clear")
white_bg_btn = Button(120, 55, 110, 35, (240, 240, 240), "White BG")
transparent_bg_btn = Button(240, 55, 130, 35, (100, 100, 100), "Transparent")
color_numbers_btn = Button(380, 55, 140, 35, (100, 180, 220), "Color Numbers")
palette_toggle_btn = Button(530, 55, 90, 35, (150, 100, 200), "Palette")

# Set initial active states
transparent_bg_btn.active = True

toolbar_buttons = [clear_btn, white_bg_btn, transparent_bg_btn, 
                   color_numbers_btn, palette_toggle_btn]

# Create sliders
brush_slider = Slider(680, 60, 150, 20, 1, 40, brush_thickness, "Brush")
eraser_slider = Slider(900, 60, 150, 20, 10, 120, eraser_thickness, "Eraser")
sliders = [brush_slider, eraser_slider]

# Initialize MediaPipe Hands
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1) as hands:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        # Decrease click cooldown
        if click_cooldown > 0:
            click_cooldown -= 1
        
        # Reset hover states
        for btn in toolbar_buttons:
            btn.hover = False
        for slider in sliders:
            slider.hover = False
        
        ui_interaction = False
        color_palette_buttons = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                
                gesture = get_gesture(hand_landmarks)
                x, y = get_finger_tip_position(hand_landmarks, frame.shape)
                
                if gesture == "clear":
                    canvas = np.zeros_like(frame)
                    cv2.putText(frame, "CANVAS CLEARED!", (450, 360), 
                               cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 255, 100), 4)
                    prev_x, prev_y = None, None
                    
                elif gesture == "erase":
                    cv2.circle(frame, (x, y), eraser_thickness//2, (255, 100, 100), 3)
                    if prev_x is not None and prev_y is not None and not ui_interaction:
                        cv2.line(canvas, (prev_x, prev_y), (x, y), 
                                (0, 0, 0), eraser_thickness)
                    prev_x, prev_y = x, y
                    
                elif gesture == "draw":
                    # Check toolbar interactions
                    for btn in toolbar_buttons:
                        if btn.is_over(x, y):
                            btn.hover = True
                            if click_cooldown == 0:
                                if btn == clear_btn:
                                    canvas = np.zeros_like(frame)
                                elif btn == white_bg_btn:
                                    background_mode = "white"
                                    white_bg_btn.active = True
                                    transparent_bg_btn.active = False
                                    color_numbers_btn.active = False
                                elif btn == transparent_bg_btn:
                                    background_mode = "transparent"
                                    transparent_bg_btn.active = True
                                    white_bg_btn.active = False
                                    color_numbers_btn.active = False
                                elif btn == color_numbers_btn:
                                    background_mode = "color_by_numbers"
                                    color_numbers_btn.active = True
                                    white_bg_btn.active = False
                                    transparent_bg_btn.active = False
                                elif btn == palette_toggle_btn:
                                    show_color_palette = not show_color_palette
                                    palette_toggle_btn.active = show_color_palette
                                
                                click_cooldown = click_delay
                            ui_interaction = True
                    
                    # Check slider interactions
                    for slider in sliders:
                        if slider.is_over(x, y):
                            slider.hover = True
                            slider.dragging = True
                            slider.update(x)
                            if slider == brush_slider:
                                brush_thickness = int(slider.current_val)
                            elif slider == eraser_slider:
                                eraser_thickness = int(slider.current_val)
                            ui_interaction = True
                    
                    # Check color palette
                    if show_color_palette:
                        color_palette_buttons = draw_color_palette(frame, 1120, 110)
                        for btn in color_palette_buttons:
                            if btn.is_over(x, y):
                                btn.hover = True
                                if click_cooldown == 0:
                                    draw_color = btn.color
                                    click_cooldown = click_delay
                                ui_interaction = True
                    
                    # Draw on canvas if not interacting with UI
                    if not ui_interaction:
                        cv2.circle(frame, (x, y), brush_thickness, draw_color, -1)
                        if prev_x is not None and prev_y is not None:
                            cv2.line(canvas, (prev_x, prev_y), (x, y), 
                                    draw_color, brush_thickness)
                        prev_x, prev_y = x, y
                    else:
                        cv2.circle(frame, (x, y), 8, (100, 255, 255), -1)
                        prev_x, prev_y = None, None
                else:
                    prev_x, prev_y = None, None
                    for slider in sliders:
                        slider.dragging = False
        else:
            prev_x, prev_y = None, None
            for slider in sliders:
                slider.dragging = False
        
        # Apply background mode
        if background_mode == "white":
            white_bg = np.ones_like(frame) * 255
            display_canvas = cv2.addWeighted(white_bg, 1, canvas, 1, 0)
            combined = cv2.addWeighted(frame, 0.3, display_canvas, 0.7, 0)
        elif background_mode == "color_by_numbers":
            template_display = cv2.addWeighted(color_by_numbers_template, 0.8, canvas, 0.5, 0)
            combined = cv2.addWeighted(frame, 0.2, template_display, 0.8, 0)
        else:  # transparent
            combined = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)
        
        # Draw UI elements
        draw_main_toolbar(combined, toolbar_buttons, sliders)
        
        if show_color_palette:
            color_palette_buttons = draw_color_palette(combined, 1120, 110)
        
        # Current color indicator
        cv2.rectangle(combined, (1100, 20), (1150, 70), draw_color, -1)
        cv2.rectangle(combined, (1100, 20), (1150, 70), (255, 255, 255), 2)
        cv2.putText(combined, "Color", (1105, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.imshow('Hand Drawing Studio', combined)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite('drawing.png', canvas)
            print("Drawing saved!")

cap.release()
cv2.destroyAllWindows()