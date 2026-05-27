import cv2
import numpy as np
from keras.models import load_model
from collections import deque
import numpy as np
import cv2
from utils import N_FRAMES,GESTURES, get_hand_data
from keras.models import Sequential

model = load_model('./models/trained_model.keras')

def predict_gesture(sequence, model: Sequential):
    model_input = np.expand_dims(np.array(sequence), axis=0)
    
    predictions = model(model_input, training=False).numpy()
    
    y_pred = np.argmax(predictions)
    
    gesture_predicted = GESTURES[y_pred]
    probability = predictions[0][y_pred]*100
    
    return gesture_predicted, probability

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
sequence = deque(maxlen=N_FRAMES)
frame_idx = 0
prev_wrist = None

def is_moving(sequence, threshold=10):
    max_variance = np.max(np.var(sequence, axis=0))
    return max_variance > threshold

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    frame_data, prev_wrist = get_hand_data(frame, frame_idx, prev_wrist)
    sequence.append(frame_data)
    
    if len(sequence) == N_FRAMES:
        if is_moving(sequence):
            gesture, prob = predict_gesture(sequence, model)
            cv2.putText(frame, f'Gesture: {gesture} | Probability: {prob:3f}%',
                        (15, 30), cv2.LINE_AA, 0.8, (0,0,0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, 'No movement', (50, 30), cv2.LINE_AA, 0.8, (0,0,0), 2, cv2.LINE_AA)
    
    cv2.imshow('', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    frame_idx += 1
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
