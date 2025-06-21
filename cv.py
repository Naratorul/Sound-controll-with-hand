
import cv2
import mediapipe as mp
import numpy as np
import time
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
from ctypes import cast, POINTER
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
face_detection = mp_face.FaceDetection(min_detection_confidence=0.7)
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volume_range = volume.GetVolumeRange()
min_vol = volume_range[0]
max_vol = volume_range[1]
cap = cv2.VideoCapture(0)
prev_time = 0
while True:
    success, img = cap.read()
    h, w, c = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results_hands = hands.process(img_rgb)
    results_face = face_detection.process(img_rgb)
    if results_face.detections:
        for detection in results_face.detections:
            bboxC = detection.location_data.relative_bounding_box
            bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                   int(bboxC.width * w), int(bboxC.height * h)
            cv2.rectangle(img, bbox, (0, 255, 0), 2)
            cv2.putText(img, 'Face', (bbox[0], bbox[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 
    if results_hands.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            hand_label = results_hands.multi_handedness[idx].classification[0].label
            if hand_label == 'Left':
                hand_label = 'Left'
            elif hand_label == 'Right':
                hand_label = 'Right'
            cv2.putText(img, f'Hand: {hand_label}', (10, 50 + idx * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
            index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
            length = np.hypot(index_x - thumb_x, index_y - thumb_y)
            vol = np.interp(length, [30, 300], [min_vol, max_vol])
            volume.SetMasterVolumeLevel(vol, None)
            cv2.putText(img, f'Volume: {int(np.interp(vol, [min_vol, max_vol], [0, 100]))}%', (50, 100 + idx * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.line(img, (thumb_x, thumb_y), (index_x, index_y), (0, 255, 0), 3)
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(img, f'FPS: {int(fps)}', (w - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
