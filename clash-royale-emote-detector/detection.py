# Media pipe holistic model
import mediapipe as mp
import cv2

# Importaciones
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pandas as pd
import numpy as np
import time

# Read the CSV file (After first run this will be used)
df = pd.read_csv('processed_emotes.csv')

# # Define the columns for the DataFrame (After first run this wont be used)
# columns = ["lhg", "rhg", "browDownLeft",
# "browDownRight",
# "browInnerUp",
# "browOuterUpLeft",
# "browOuterUpRight",
# "cheekPuff",
# "cheekSquintLeft",
# "cheekSquintRight",
# "eyeBlinkLeft",
# "eyeBlinkRight",
# "eyeLookDownLeft",
# "eyeLookDownRight",
# "eyeLookInLeft",
# "eyeLookInRight",
# "eyeLookOutLeft",
# "eyeLookOutRight",
# "eyeLookUpLeft",
# "eyeLookUpRight",
# "eyeSquintLeft",
# "eyeSquintRight",
# "eyeWideLeft",
# "eyeWideRight",
# "jawForward",
# "jawLeft",
# "jawOpen",
# "jawRight",
# "mouthClose",
# "mouthDimpleLeft",
# "mouthDimpleRight",
# "mouthFrownLeft",
# "mouthFrownRight",
# "mouthFunnel",
# "mouthLeft",
# "mouthLowerDownLeft",
# "mouthLowerDownRight",
# "mouthPressLeft",
# "mouthPressRight",
# "mouthPucker",
# "mouthRight",
# "mouthRollLower",
# "mouthRollUpper",
# "mouthShrugLower",
# "mouthShrugUpper",
# "mouthSmileLeft",
# "mouthSmileRight",
# "mouthStretchLeft",
# "mouthStretchRight",
# "mouthUpperUpLeft",
# "mouthUpperUpRight",
# "noseSneerLeft",
# "noseSneerRight",
# "tongueOut", "ll1x", "ll1y", "ll2x", "ll2y", "ll3x", "ll3y", "ll4x", "ll4y", "ll5x", "ll5y", "ll6x", "ll6y",
#             "ll7x", "ll7y", "ll8x", "ll8y", "ll9x", "ll9y", "ll10x", "ll10y", "ll11x", "ll11y", "ll12x", "ll12y", "ll13x",
#             "ll13y", "ll14x", "ll14y", "ll15x", "ll15y", "ll16x", "ll16y", "ll17x", "ll17y", "ll18x", "ll18y", "ll19x", "ll19y",
#             "ll20x", "ll20y", "rl1x", "rl1y", "rl2x", "rl2y", "rl3x", "rl3y", "rl4x", "rl4y", "rl5x", "rl5y", "rl6x", "rl6y",
#             "rl7x", "rl7y", "rl8x", "rl8y", "rl9x", "rl9y", "rl10x", "rl10y", "rl11x", "rl11y",
#             "rl12x", "rl12y", "rl13x", "rl13y", "rl14x", "rl14y", "rl15x", "rl15y", "rl16x", "rl16y", "rl17x", "rl17y",
#             "rl18x", "rl18y", "rl19x", "rl19y", "rl20x", "rl20y", "emote_label"]
# df = pd.DataFrame(columns=columns)

# Model Setup
MODEL_PATH = "./gesture_recognition.task"
FACE_MODEL_PATH = "./face_recognition.task"

# Hand Gesture Recognizer
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.GestureRecognizerOptions(
    base_options=base_options,
    num_hands=2,
    running_mode=vision.RunningMode.VIDEO
)

detector = vision.GestureRecognizer.create_from_options(options)

# Face Landmark Detector
face_base_options = python.BaseOptions(model_asset_path=FACE_MODEL_PATH)
face_options = vision.FaceLandmarkerOptions(
    base_options=face_base_options,
    running_mode=vision.RunningMode.VIDEO,
    output_face_blendshapes=True
)
face_detector = vision.FaceLandmarker.create_from_options(face_options)

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

# Functions to capture image and process the landmarks it

# Label the gestures with a number
gesture_labels = [
    "None",
    "Closed_Fist",
    "Open_Palm",
    "Pointing_Up",
    "Thumbs_Down",
    "Thumbs_Up",
    "Victory",
    "ILoveYou"
]
def process_hand_gestures(hand_label):
    return gesture_labels.index(hand_label) if hand_label in gesture_labels else 0

# Camera Live Recording
cap = cv2.VideoCapture(0)
timestamp = 0

# Emote List
emotes = {
    0: "Thumbs_Up_Goblin",
    1: "Yawning_Princess",
    2: "Goblin_Peace_Sign",
    3: "Angry_Barb",
    4: "Magician_Fire",
    5: "Surprised_Bandit",
}

# So the screen doesnt freezes
capture_requested = False
capture_time = 0
delay_seconds = 2
emote_to_capture = None

while cap.isOpened():
    # Acts upon key pressing
    key = cv2.waitKey(1) & 0xFF

    # Read frame
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    timestamp = int(time.time() * 1000)

    # Hand Gesture Recognition
    result = detector.recognize_for_video(mp_image, timestamp)
    if result.hand_landmarks:
        for h, hand_landmarks in enumerate(result.hand_landmarks):
            draw_hand_on_frame(frame, hand_landmarks)   

            if result.handedness and len(result.handedness) > h:
                hand_label = result.handedness[h][0].category_name 
                hand_score = result.handedness[h][0].score

            if result.gestures and len(result.gestures) > h:
                gesture = result.gestures[h][0]
                gesture_name = gesture.category_name
                confidence = gesture.score

                if hand_label == "Left":
                    hand_label = "Right"
                else:
                    hand_label = "Left"

                text = f"{hand_label}: {gesture_name} ({confidence:.2f})"

                height, width, _ = frame.shape
                x = int(hand_landmarks[0].x * width)
                y = int(hand_landmarks[0].y * height) - 20

                cv2.putText(
                    frame,
                    text,
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

    # Face Blendshapes Detection
    face_result = face_detector.detect_for_video(mp_image, timestamp)
    if face_result and face_result.face_blendshapes and len(face_result.face_blendshapes) > 0:
        blendshapes = face_result.face_blendshapes[0]
        for blendshape in blendshapes:
            if blendshape.score > 0.5:
                cv2.putText(
                    frame,
                    f"{blendshape.category_name}: {blendshape.score:.2f}",
                    (10, 30 + 30 * blendshapes.index(blendshape)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA
                )

    # Check if capture is requested
    if key in [ord(str(i)) for i in range(8)]:
        capture_requested = True
        capture_time = time.time()
        emote_to_capture = chr(key)
        print("Emote now...")

    # Capture Image on Specific Gesture and Process Data
    if (
            capture_requested and
            result.hand_landmarks and
            face_result.face_blendshapes and
            time.time() - capture_time >= delay_seconds
    ):

        data_row = np.zeros(135)  # 2 for hand gestures, 52 for blendshapes, 80 for hand landmarks + 1 for label

        # Process hand gestures
        for h, hand_landmarks in enumerate(result.hand_landmarks):
            if result.handedness and len(result.handedness) > h:
                hand_label = result.handedness[h][0].category_name

            if result.gestures and len(result.gestures) > h:
                gesture = result.gestures[h][0]
                gesture_name = gesture.category_name

                if hand_label == "Left":
                    data_row[1] = process_hand_gestures(gesture_name)
                else:
                    data_row[0] = process_hand_gestures(gesture_name)

        # Process face blendshapes
        blendshapes = face_result.face_blendshapes[0]
        for blendshape in blendshapes:
            idx = blendshapes.index(blendshape) - 1

            #Skip Neutral
            if blendshape.category_name == "neutral":
                continue

            if blendshape.score > 0.5:
                data_row[2 + idx] = blendshape.score
            else:
                # I do this so low values don't interfere with training
                data_row[2 + idx] = 0.0

        # Process hand landmarks positions relative to the palm
        for h, hand_landmarks in enumerate(result.hand_landmarks):
            if not result.handedness or len(result.handedness) <= h:
                continue

            hand_label = result.handedness[h][0].category_name

            if hand_label == "Left":
                offset = 94
            elif hand_label == "Right":
                offset = 54
            else:
                continue

            palm_x = hand_landmarks[0].x
            palm_y = hand_landmarks[0].y

            write_index = 0

            for l, landmark in enumerate(hand_landmarks):
                if l == 0:
                    continue

                data_row[offset + write_index] = palm_x - landmark.x
                data_row[offset + write_index + 1] = palm_y - landmark.y

                write_index += 2

        # Append the label for the emote (the key pressed)
        data_row[-1] = int(emote_to_capture)

        # Save the data row to the DataFrame
        df.loc[len(df)] = data_row

        # Confirm Capture
        print(f"Captured Emote: {emotes[int(emote_to_capture)]}")
        print(f"Hand Gestures: Left - {data_row[0]}, Right - {data_row[1]}")
        print(f"Face Blendshapes: {data_row[2:54]}")
        print(f"Hand Landmarks (relative to palm): {data_row[54:134]}")

        # Reset capture request
        capture_requested = False
        emote_to_capture = None

    # Display the resulting frame
    cv2.imshow("Hand and Face Tracking", frame)
    if key == 27:
        # Save CSV
        df.to_csv('processed_emotes.csv', index=False)
        break

cap.release()
cv2.destroyAllWindows()
