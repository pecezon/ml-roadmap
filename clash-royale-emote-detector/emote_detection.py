# Media pipe holistic model
import mediapipe as mp
import cv2

# Importaciones
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import joblib
import pandas as pd
import pygame

# Initialize Pygame Mixer for Audio Playback
pygame.mixer.init()

# Trained ML Model
model, feature_names = joblib.load('emote_detection_pipeline_v2.pkl')

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
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (5, 9), (9, 13), (13, 17)  # Palm
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

# Label the gestures with a number
gesture_labels = [
    "None",
    "Closed_Fist",
    "Open_Palm",
    "Pointing_Up",
    "Thumb_Down",
    "Thumb_Up",
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
    6: "Nose_Picking_Barb",
    7: "Hog_Rider_Kiss",
    8: "Angry_Giant",
    9: "Happy_Royale_Ghost",
}

# Emote Video Files
emote_videos = {
    0: "emote_videos/goblin_thumbs_up.mp4",
    1: "emote_videos/yawning_princess.mp4",
    2: "emote_videos/goblin_peace.mp4",
    3: "emote_videos/angry_barb.mp4",
    4: "emote_videos/magician_fire.mp4",
    5: "emote_videos/surprised_bandit.mp4",
    6: "emote_videos/nose_picking_barb.mp4",
    7: "emote_videos/hog_rider_kiss.mp4",
    8: "emote_videos/angry_giant.mp4",
    9: "emote_videos/happy_royale_ghost.mp4",
}

# Emote Audio Files
emote_audios = {
    0: "emote_audios/goblin_thumbs_up.wav",
    1: "emote_audios/yawning_princess.wav",
    2: "emote_audios/goblin_peace.wav",
    3: "emote_audios/angry_barb.wav",
    4: "emote_audios/magician_fire.wav",
    5: "emote_audios/surprised_bandit.wav",
    6: "emote_audios/nose_picking_barb.wav",
    7: "emote_audios/hog_rider_kiss.wav",
    8: "emote_audios/angry_giant.wav",
    9: "emote_audios/happy_royale_ghost.wav",
}


current_emote_cap = None
current_emote_id = None
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

    if (
            result.hand_landmarks and
            face_result.face_blendshapes
    ):

        data_row = np.zeros(134)  # 2 for hand gestures, 52 for blendshapes, 80 for hand landmarks

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

            # Skip Neutral
            if blendshape.category_name == "_neutral":
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

         # Predict Emote
        input_df = pd.DataFrame([data_row], columns=feature_names)
        predicted_emote = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]
        emote_name = emotes.get(predicted_emote, "Unknown")

        #Video Playback
        if predicted_emote != current_emote_id:
            current_emote_id = predicted_emote

            if current_emote_cap:
                current_emote_cap.release()

            video_emote_path = emote_videos.get(predicted_emote)
            audio_emote_path = emote_audios.get(predicted_emote)

            if video_emote_path:
                current_emote_cap = cv2.VideoCapture(video_emote_path)

            if audio_emote_path:
                pygame.mixer.music.load(audio_emote_path)
                pygame.mixer.music.play()

        # Text With the prediction
        cv2.putText(
            frame,
            f"Predicted Emote: {emote_name}. Accuracy: {proba[predicted_emote]:.2f}",
            (10, frame.shape[0] - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

    # Display Emote Video
    if current_emote_cap:
        ret_emote, emote_frame = current_emote_cap.read()

        if not ret_emote:
            # Restart animation when it finishes
            current_emote_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret_emote, emote_frame = current_emote_cap.read()

        if ret_emote:
            # Resize emote
            emote_frame = cv2.resize(emote_frame, (200, 200))

            h, w, _ = frame.shape

            # Top-right corner
            y1, y2 = 10, 210
            x1, x2 = w - 210, w - 10

            frame[y1:y2, x1:x2] = emote_frame

    # Display the resulting frame
    cv2.imshow("Hand and Face Tracking", frame)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
