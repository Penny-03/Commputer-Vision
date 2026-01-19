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

# ===================== UI CLASSES =====================
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
        cv2.rectangle(frame, (self.x, self.y), (self.x + self.w, self.y + self.h),
                      self.color, -1)

        border_thickness = 4 if self.active else 2
        cv2.rectangle(frame, (self.x, self.y), (self.x + self.w, self.y + self.h),
                      (255, 255, 255), border_thickness)

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
        cv2.rectangle(frame, (self.x, self.y), (self.x + self.w, self.y + self.h),
                      (100, 100, 100), -1)

        value_ratio = (self.current_val - self.min_val) / (self.max_val - self.min_val)
        knob_x = int(self.x + value_ratio * self.w)

        cv2.rectangle(frame, (self.x, self.y), (knob_x, self.y + self.h), (0, 200, 0), -1)

        cv2.circle(frame, (knob_x, self.y + self.h // 2), 12, (255, 255, 255), -1)
        cv2.circle(frame, (knob_x, self.y + self.h // 2), 12, (0, 0, 0), 2)

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
        distance = np.sqrt((x - knob_x) ** 2 + (y - self.y - self.h // 2) ** 2)
        return distance <= 15 or (self.x <= x <= self.x + self.w and
                                 self.y <= y <= self.y + self.h)

# ===================== UI ELEMENTS =====================
color_buttons = [
    Button(10, 150, 50, 50, (0, 0, 255), "Red"),
    Button(70, 150, 50, 50, (0, 255, 0), "Green"),
    Button(130, 150, 50, 50, (255, 0, 0), "Blue"),
    Button(190, 150, 50, 50, (0, 255, 255), "Yellow"),
    Button(250, 150, 50, 50, (255, 0, 255), "Pink"),
    Button(310, 150, 50, 50, (255, 255, 255), "White"),
    Button(370, 150, 50, 50, (0, 0, 0), "Black"),
]

clear_button = Button(10, 220, 100, 50, (200, 200, 200), "Clear")
eraser_button = Button(120, 220, 100, 50, (150, 150, 150), "Eraser")

brush_slider = Slider(10, 300, 200, 20, 1, 30, brush_thickness, "Brush")
eraser_slider = Slider(10, 350, 200, 20, 10, 100, eraser_thickness, "Eraser")

# ===================== HAND HELPERS =====================
def landmark_to_pixel(hand_landmarks, idx, frame_shape):
    lm = hand_landmarks.landmark[idx]
    h, w, _ = frame_shape
    return int(lm.x * w), int(lm.y * h)

def get_index_tip_position(hand_landmarks, frame_shape):
    return landmark_to_pixel(hand_landmarks, 8, frame_shape)

def is_finger_up(hand_landmarks, finger_tip_id, finger_pip_id):
    tip = hand_landmarks.landmark[finger_tip_id]
    pip = hand_landmarks.landmark[finger_pip_id]
    return tip.y < pip.y

def is_thumb_middle_pinch(hand_landmarks, frame_shape, pinch_thresh_px=40):
    tx, ty = landmark_to_pixel(hand_landmarks, 4, frame_shape)   # thumb tip
    mx, my = landmark_to_pixel(hand_landmarks, 12, frame_shape)  # middle tip
    dist = np.hypot(tx - mx, ty - my)
    return dist < pinch_thresh_px

def are_index_middle_tips_close(hand_landmarks, frame_shape, close_thresh_px=35):
    ix, iy = landmark_to_pixel(hand_landmarks, 8, frame_shape)    # index tip
    mx, my = landmark_to_pixel(hand_landmarks, 12, frame_shape)   # middle tip
    dist = np.hypot(ix - mx, iy - my)
    return dist < close_thresh_px

def get_gesture(hand_landmarks, frame_shape):
    """
    RULES:
      - ERASE: index + middle tips close
      - PAINT: index up + thumb-middle pinch
      - else: none
    """
    if are_index_middle_tips_close(hand_landmarks, frame_shape, close_thresh_px=35):
        return "erase"

    index_up = is_finger_up(hand_landmarks, 8, 6)
    if index_up and is_thumb_middle_pinch(hand_landmarks, frame_shape, pinch_thresh_px=40):
        return "paint"

    return "none"

# ===================== UI LOGIC =====================
def check_ui_interaction(x, y):
    global draw_color, brush_thickness, eraser_thickness, canvas

    for btn in color_buttons:
        if btn.is_clicked(x, y):
            draw_color = btn.color
            for b in color_buttons:
                b.active = False
            btn.active = True
            return True

    if clear_button.is_clicked(x, y):
        canvas[:] = 0
        return True

    # Manual eraser button (optional to keep)
    if eraser_button.is_clicked(x, y):
        return True

    if brush_slider.is_clicked(x, y):
        brush_slider.dragging = True
        brush_slider.update(x)
        brush_thickness = int(brush_slider.current_val)
        return True

    if eraser_slider.is_clicked(x, y):
        eraser_slider.dragging = True
        eraser_slider.update(x)
        eraser_thickness = int(eraser_slider.current_val)
        return True

    return False

def draw_ui(frame):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 140), (450, 400), (50, 50, 50), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.putText(frame, "Tools", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    for btn in color_buttons:
        btn.draw(frame)

    clear_button.draw(frame)
    eraser_button.draw(frame)

    brush_slider.draw(frame)
    eraser_slider.draw(frame)

    cv2.putText(frame, "Current:", (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.rectangle(frame, (100, 420), (150, 450), draw_color, -1)
    cv2.rectangle(frame, (100, 420), (150, 450), (255, 255, 255), 2)

    cv2.putText(frame, "Paint: Index up + Thumb&Middle pinch", (10, 470),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "Erase: Index&Middle tips close", (10, 490),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "Clear: use Clear button | Else: do nothing", (10, 510),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# ===================== MAIN =====================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

ret, frame = cap.read()
if ret:
    canvas = np.zeros_like(frame)

color_buttons[1].active = True  # Default green

click_cooldown = 0
click_delay = 12

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if click_cooldown > 0:
            click_cooldown -= 1

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                gesture = get_gesture(hand_landmarks, frame.shape)
                x, y = get_index_tip_position(hand_landmarks, frame.shape)

                if gesture == "paint":
                    ui_clicked = check_ui_interaction(x, y)
                    if ui_clicked:
                        if click_cooldown == 0:
                            click_cooldown = click_delay
                        prev_x, prev_y = None, None
                    else:
                        brush_thickness = int(brush_slider.current_val)
                        cv2.circle(frame, (x, y), brush_thickness, draw_color, -1)
                        if prev_x is not None and prev_y is not None:
                            cv2.line(canvas, (prev_x, prev_y), (x, y), draw_color, brush_thickness)
                        prev_x, prev_y = x, y

                elif gesture == "erase":
                    eraser_thickness = int(eraser_slider.current_val)
                    cv2.circle(frame, (x, y), eraser_thickness // 2, (0, 0, 255), 2)
                    if prev_x is not None and prev_y is not None:
                        cv2.line(canvas, (prev_x, prev_y), (x, y), (0, 0, 0), eraser_thickness)
                    prev_x, prev_y = x, y

                else:
                    prev_x, prev_y = None, None
                    brush_slider.dragging = False
                    eraser_slider.dragging = False

        else:
            prev_x, prev_y = None, None
            brush_slider.dragging = False
            eraser_slider.dragging = False

        combined = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)
        draw_ui(combined)

        cv2.imshow("Hand Drawing", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            cv2.imwrite("drawing.png", canvas)
            print("Drawing saved as 'drawing.png'")

cap.release()
cv2.destroyAllWindows()

