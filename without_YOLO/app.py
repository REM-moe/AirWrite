import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Load pre-trained handwriting recognition model
model = keras.models.load_model('handwriting_model.keras')

# Global variables
drawing_path = []
is_drawing = False
expression = []
last_symbol_time = 0
SYMBOL_TIMEOUT = 1.0  # seconds to wait before considering a new symbol

def recognize_digit(image):
    # Resize and normalize the image
    resized = cv2.resize(image, (28, 28))
    normalized = resized / 255.0
    # Predict using the model
    prediction = model.predict(normalized.reshape(1, 28, 28, 1), verbose=0)
    return np.argmax(prediction)

def recognize_operator(points):
    # Simple operator recognition based on shape analysis
    if len(points) < 2:
        return None
    
    # Convert points to numpy array
    points = np.array(points)
    
    # Calculate bounding box
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    width = x_max - x_min
    height = y_max - y_min
    
    # Check if it's a plus sign (roughly equal horizontal and vertical lines intersecting)
    if 0.8 < width/height < 1.2:
        return '+'
    
    # Check if it's a minus sign (horizontal line wider than tall)
    if width > height * 2:
        return '-'
    
    return None

def evaluate_expression(expr):
    if not expr:
        return "No expression"
    
    try:
        # Convert expression list to string
        expr_str = ''.join(str(x) for x in expr)
        # Evaluate the expression
        result = eval(expr_str)
        return f"{expr_str} = {result}"
    except:
        return "Invalid expression"

def process_frame(frame):
    global drawing_path, is_drawing, expression, last_symbol_time
    
    if frame is None:
        return None
    
    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)
    
    # Process the frame with MediaPipe Hands
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the tip of the index finger
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            h, w, _ = frame.shape
            x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            
            # Check if the index finger is raised (drawing mode)
            if hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y:
                is_drawing = True
                if (x, y) not in drawing_path:  # Prevent duplicate points
                    drawing_path.append((x, y))
            else:
                is_drawing = False
            
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    # Draw the air writing path
    if len(drawing_path) > 1:
        points = np.array(drawing_path, dtype=np.int32)
        cv2.polylines(frame, [points], False, (0, 255, 0), 2)
    
    # Draw current expression
    expr_text = ' '.join(str(x) for x in expression)
    cv2.putText(frame, f'Expression: {expr_text}', (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # If an expression exists, show the result
    if expression:
        result = evaluate_expression(expression)
        cv2.putText(frame, f'Result: {result}', (50, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return frame

def main():
    global drawing_path, expression, last_symbol_time
    
    # Initialize video capture with default camera (0)
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
            
            processed_frame = process_frame(frame)
            if processed_frame is None:
                continue
            
            current_time = cv2.getTickCount() / cv2.getTickFrequency()
            
            # If drawing is complete and enough time has passed, recognize the symbol
            if not is_drawing and len(drawing_path) > 10 and (current_time - last_symbol_time) > SYMBOL_TIMEOUT:
                # Create a blank image and draw the path
                digit_image = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                points = np.array(drawing_path, dtype=np.int32)
                cv2.polylines(digit_image, [points], False, 255, 2)
                
                # Try to recognize operator first
                operator = recognize_operator(drawing_path)
                if operator:
                    expression.append(operator)
                else:
                    # If not an operator, recognize as digit
                    digit = recognize_digit(digit_image)
                    expression.append(digit)
                
                # Update last symbol time
                last_symbol_time = current_time
                # Clear the drawing path
                drawing_path = []
            
            cv2.imshow('Math Expression Recognition', processed_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Clear expression with 'c' key
                expression = []
                drawing_path = []
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()

if __name__ == '__main__':
    main()