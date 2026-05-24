import cv2
import numpy as np
import time
from pathlib import Path
from utils import N_FRAMES, N_SEQUENCES, GESTURES, get_hand_data

gesture = GESTURES[3]
path = Path('dataset') / gesture
path.mkdir(parents=True, exist_ok=True)

existing = sorted(path.glob("*-sequence.npy"))
start_seq = len(existing)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
frame_idx = 0
stop = False

if start_seq != N_SEQUENCES:
    for seq in range(start_seq, N_SEQUENCES):
        if stop:
            break
        time_start = time.time()
        
        #Espera entre secuencia
        while time.time() - time_start < 0.5:
            ret, frame = cap.read()
            if not ret:
                break
            
            cv2.putText(frame, f'gesture: {gesture} | seq: {seq+1}/{N_SEQUENCES}', (10, 40), cv2.LINE_AA, 1, (0, 255, 0), 2)
            
            cv2.imshow('', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop = True
                break
        
        if stop: 
            break
        
        #Grabar secuencia
        sequence_data = []
        prev_wrist = None
        for _ in range(N_FRAMES):
            ret, frame = cap.read()
            if not ret:
                stop = True
                break
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            frame_data, prev_wrist = get_hand_data(frame, frame_idx, prev_wrist)
            
            sequence_data.append(frame_data)
            
            cv2.putText(frame, f'RECORD: {gesture}', (20, 40), cv2.LINE_AA, 1, (255, 0, 0), 2)
            
            cv2.imshow('', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            frame_idx += 1  
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop = True
                break
        
        if len(sequence_data) == N_FRAMES:
            np.save(f'./dataset/{gesture}/{seq}-sequence.npy', np.array(sequence_data))

cap.release()
cv2.destroyAllWindows()
