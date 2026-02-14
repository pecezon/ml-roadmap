# Media pipe holistic model
import mediapipe as mp
import cv2
import pygame

# Importaciones
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
pygame.mixer.init()

# Lista de sonidos
sounds = [
    pygame.mixer.Sound("./sounds/d#.mp3"),   # 0: Meñique Izquierdo
    pygame.mixer.Sound("./sounds/e.mp3"),   # 1: Índice Izquierdo
    pygame.mixer.Sound("./sounds/ed.mp3"),  # 2: Medio Izquierdo
    pygame.mixer.Sound("./sounds/d#d.mp3"), # 3: Anular Izquierdo
    pygame.mixer.Sound("./sounds/g.mp3"),   # 4: Meñique Derecho
    pygame.mixer.Sound("./sounds/c#.mp3"),  # 5: Índice Derecho
    pygame.mixer.Sound("./sounds/c#d.mp3"), # 6: Medio Derecho
    pygame.mixer.Sound("./sounds/a.mp3")   # 7: Anular Derecho
]

# Function to detect finger down
def is_finger_down(landmarks, finger_tip, finger_mcp):
    return landmarks[finger_tip].y > landmarks[finger_mcp].y

# Model Setup
MODEL_PATH = "./hand_landmarker.task"

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    running_mode=vision.RunningMode.VIDEO
)

detector = vision.HandLandmarker.create_from_options(options)

# Drawing hands utilities
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),    # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),    # Index
    (0, 9), (9, 10), (10, 11), (11, 12), # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17)          # Palm
]

def draw_hand_on_frame(frame, hand_landmarks):
    height, width, _ = frame.shape
    
    # Draw Connections (Lines)
    for connection in HAND_CONNECTIONS:
        start_idx = connection[0]
        end_idx = connection[1]
        
        # Convert normalized coordinates (0-1) to pixel coordinates
        start_point = (int(hand_landmarks[start_idx].x * width), 
                       int(hand_landmarks[start_idx].y * height))
        end_point = (int(hand_landmarks[end_idx].x * width), 
                     int(hand_landmarks[end_idx].y * height))
        
        cv2.line(frame, start_point, end_point, (255, 255, 255), 2)

    # Draw Landmarks (Dots)
    for landmark in hand_landmarks:
        pixel_x = int(landmark.x * width)
        pixel_y = int(landmark.y * height)
        cv2.circle(frame, (pixel_x, pixel_y), 5, (0, 255, 0), -1)

# Camera Live Recording
cap = cv2.VideoCapture(0)

timestamp = 0

finger_state = [False] * 8 

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    timestamp += 1
    result = detector.detect_for_video(mp_image, timestamp)

    if result.hand_landmarks:
        for h, hand_landmarks in enumerate(result.hand_landmarks):
            draw_hand_on_frame(frame, hand_landmarks)   
            
            finger_tips = [8, 12, 16, 20]
            finger_mcp = [5, 9, 13, 17]

            for i in range(4):
                finger_index = i + (4 * h)
                is_down =  is_finger_down(hand_landmarks, finger_tips[i], finger_mcp[i])
                if is_down:
                    if not finger_state[finger_index]:
                        sounds[finger_index].play()
                        finger_state[finger_index] = True
                else:
                    finger_state[finger_index] = False



    cv2.imshow("Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
