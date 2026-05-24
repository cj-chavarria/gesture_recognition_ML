import numpy as np
import mediapipe as mp
from mediapipe.tasks.python.vision import drawing_utils, HandLandmarksConnections
from keras.models import Sequential

N_FRAMES = 60
N_SEQUENCES = 80
GESTURES = ['hello', 'no_hand', 'calling', 'pointing', 'so_so']

model_path = 'models/hand_landmarker.task'
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=model_path), running_mode=VisionRunningMode.VIDEO)

landmarker = HandLandmarker.create_from_options(options)
hand_connections = HandLandmarksConnections.HAND_CONNECTIONS


def get_hand_data(frame, frame_idx: int, prev_wrist = None) -> tuple:
    frame_data = []
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    frame_timestamp_ms = frame_idx * 33
    detection = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
    
    hand_landmarks = detection.hand_landmarks
    if hand_landmarks:
        landmarks = hand_landmarks[0]
        
        drawing_utils.draw_landmarks(frame, landmarks, hand_connections)
        
        wrist = landmarks[0]
        middle_finger_mcp = landmarks[9]
        
        hand_size_scale = np.sqrt((middle_finger_mcp.x - wrist.x)**2 + (middle_finger_mcp.y - wrist.y)**2 + (middle_finger_mcp.z - wrist.z)**2) + 1e-10
        
        for landmark in landmarks:
            norm_x = (landmark.x - wrist.x) / hand_size_scale
            norm_y = (landmark.y - wrist.y) / hand_size_scale
            frame_data.extend([norm_x, norm_y])
        
        if prev_wrist:
            delta_x = (wrist.x - prev_wrist.x) / hand_size_scale * 100
            delta_y = (wrist.y - prev_wrist.y) / hand_size_scale * 100
        else:
            delta_x, delta_y = 0, 0
        
        frame_data.extend([delta_x, delta_y])
        
        return frame_data, wrist
        
    else:
        return list(np.zeros(44)), None