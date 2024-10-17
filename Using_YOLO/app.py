import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from ultralytics import YOLO
import time

class FingerMathSystem:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        # Initialize YOLO model for finger counting
        self.yolo_model = YOLO('yolov8n.pt')  # Using nano model for speed
        
        # Load handwriting recognition model
        self.digit_model = tf.keras.models.load_model('without_YOLO/handwriting_model.keras')

        # State variables
        self.current_number = []  # Store digits for current number
        self.expression = []      # Store full expression
        self.drawing_path = []    # Store drawing points
        self.is_drawing = False
        self.operator_mode = False  # True when expecting operator input
        self.last_detection_time = time.time()
        self.detection_cooldown = 1.0  # Seconds between detections

        # Drawing settings
        self.canvas_size = (640, 480)
        self.draw_color = (0, 255, 0)
        self.draw_thickness = 2

    def count_raised_fingers(self, hand_landmarks):
        """Count number of raised fingers using MediaPipe landmarks"""
        finger_tips = [
            self.mp_hands.HandLandmark.THUMB_TIP,
            self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            self.mp_hands.HandLandmark.RING_FINGER_TIP,
            self.mp_hands.HandLandmark.PINKY_TIP
        ]
        finger_pips = [
            self.mp_hands.HandLandmark.THUMB_IP,
            self.mp_hands.HandLandmark.INDEX_FINGER_PIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
            self.mp_hands.HandLandmark.RING_FINGER_PIP,
            self.mp_hands.HandLandmark.PINKY_PIP
        ]
        
        count = 0
        for tip, pip in zip(finger_tips, finger_pips):
            if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
                count += 1
                
        return count

    def process_operator(self, num_fingers):
        """Convert finger count to operator"""
        if num_fingers == 2:
            return '+'
        elif num_fingers == 3:
            return '-'
        elif num_fingers == 4:
            return '*'
        return None

    def recognize_digit(self, image):
        """Recognize a single digit from image"""
        # Preprocess image
        resized = cv2.resize(image, (28, 28))
        normalized = resized.astype('float32') / 255.0
        
        # Make prediction
        prediction = self.digit_model.predict(
            normalized.reshape(1, 28, 28, 1), 
            verbose=0
        )
        return np.argmax(prediction)

    def format_number(self, digits):
        """Format digits as 3-digit number string"""
        if not digits:
            return "000"
        num_str = ''.join(map(str, digits))
        return num_str.zfill(3)

    def evaluate_expression(self):
        """Evaluate the current mathematical expression"""
        if len(self.expression) != 3:
            return "Incomplete expression"
            
        try:
            expr_str = ''.join(str(x) for x in self.expression)
            result = eval(expr_str)
            return f"{expr_str} = {result}"
        except:
            return "Invalid expression"

    def process_frame(self, frame):
        """Process a single frame from the camera"""
        if frame is None:
            return None

        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # MediaPipe hand detection
        results = self.hands.process(frame_rgb)
        
        # Process hand landmarks if detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )
                
                if self.operator_mode:
                    # Count fingers for operator input
                    num_fingers = self.count_raised_fingers(hand_landmarks)
                    operator = self.process_operator(num_fingers)
                    
                    if operator and (time.time() - self.last_detection_time) > self.detection_cooldown:
                        self.expression.append(operator)
                        self.operator_mode = False
                        self.last_detection_time = time.time()
                else:
                    # Track finger for number drawing
                    index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    h, w, _ = frame.shape
                    x, y = int(index_tip.x * w), int(index_tip.y * h)
                    
                    if index_tip.y < hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP].y:
                        self.is_drawing = True
                        self.drawing_path.append((x, y))
                    else:
                        self.is_drawing = False

        # Draw current path
        if len(self.drawing_path) > 1:
            points = np.array(self.drawing_path, dtype=np.int32)
            cv2.polylines(frame, [points], False, self.draw_color, self.draw_thickness)

        # Process completed digit
        if not self.is_drawing and len(self.drawing_path) > 10:
            # Create digit image
            digit_image = np.zeros(self.canvas_size[::-1], dtype=np.uint8)
            cv2.polylines(digit_image, [np.array(self.drawing_path)], False, 255, 2)
            
            # Recognize digit
            digit = self.recognize_digit(digit_image)
            self.current_number.append(digit)
            
            # Clear drawing path
            self.drawing_path = []
            
            # If we have 3 digits, add to expression
            if len(self.current_number) == 3:
                num_str = self.format_number(self.current_number)
                self.expression.append(num_str)
                self.current_number = []
                self.operator_mode = True

        # Draw UI elements
        self.draw_ui(frame)
        
        return frame

    def draw_ui(self, frame):
        """Draw UI elements on the frame"""
        # Draw current expression
        expr_str = ' '.join(str(x) for x in self.expression)
        cv2.putText(frame, f'Expression: {expr_str}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Draw current number being entered
        current_num = self.format_number(self.current_number)
        cv2.putText(frame, f'Current Number: {current_num}', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # Draw mode indicator
        mode = "OPERATOR" if self.operator_mode else "NUMBER"
        cv2.putText(frame, f'Mode: {mode}', (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw result if expression is complete
        if len(self.expression) == 3:
            result = self.evaluate_expression()
            cv2.putText(frame, f'Result: {result}', (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    def run(self):
        """Main loop for the application"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break
                
                processed_frame = self.process_frame(frame)
                if processed_frame is None:
                    continue
                
                cv2.imshow('Finger Math System', processed_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    # Clear everything
                    self.expression = []
                    self.current_number = []
                    self.drawing_path = []
                    self.operator_mode = False
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.hands.close()

if __name__ == '__main__':
    system = FingerMathSystem()
    system.run()