import cv2
import mediapipe as mp
import time
import os
from collections import deque
import math

# ================= MediaPipe Setup =================
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

face = mp_face.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.7
)

face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    max_num_faces=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ✅ FIX: Finger tip IDs
TIP_IDS = [4, 8, 12, 16, 20]

# ================= Blink Setup =================
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

BLINK_THRESHOLD = 0.21
BLINK_FRAMES = 2
blink_counter = 0
blink_cooldown = 0

# ================= Utils =================
def euclidean(p1, p2):
    return math.dist(p1, p2)

def eye_aspect_ratio(landmarks, eye):
    p1 = landmarks[eye[1]]
    p2 = landmarks[eye[5]]
    p3 = landmarks[eye[2]]
    p4 = landmarks[eye[4]]
    p5 = landmarks[eye[0]]
    p6 = landmarks[eye[3]]

    vertical1 = euclidean(p1, p2)
    vertical2 = euclidean(p3, p4)
    horizontal = euclidean(p5, p6)

    return (vertical1 + vertical2) / (2.0 * horizontal)

# ================= Age Group =================
def estimate_age_group(face_h, frame_h):
    ratio = face_h / frame_h
    if ratio > 0.45:
        return "0 to 10"
    elif ratio > 0.35:
        return "11 to 20"
    elif ratio > 0.25:
        return "21 to 35"
    elif ratio > 0.18:
        return "36 to 55"
    else:
        return "56+"

# ================= Hand State =================
def create_hand_state():
    return {
        "finger_buffer": deque(maxlen=5),
        "last_finger": None
    }

hand_states = {
    "Left": create_hand_state(),
    "Right": create_hand_state()
}

FINGER_TO_ASL = {
    0: "A",
    1: "D",
    2: "V",
    3: "W",
    4: "B",
    5: "I"
}

current_word = ""
STABLE_THRESHOLD = 4

# ================= Storage =================
os.makedirs("Captured", exist_ok=True)

cap = cv2.VideoCapture(0)

# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_results = face.process(rgb)
    mesh_results = face_mesh.process(rgb)
    hand_results = hands.process(rgb)

    # ================= FACE BOX + AGE =================
    if face_results.detections:
        for det in face_results.detections:
            box = det.location_data.relative_bounding_box

            x1 = int(box.xmin * w)
            y1 = int(box.ymin * h)
            x2 = int((box.xmin + box.width) * w)
            y2 = int((box.ymin + box.height) * h)

            face_h = y2 - y1
            age_range = estimate_age_group(face_h, h)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(
                frame,
                f"Age Group: {age_range}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

    # ================= BLINK DETECTION =================
    if mesh_results.multi_face_landmarks:
        landmarks = mesh_results.multi_face_landmarks[0].landmark
        points = [(int(l.x * w), int(l.y * h)) for l in landmarks]

        ear = (
            eye_aspect_ratio(points, LEFT_EYE) +
            eye_aspect_ratio(points, RIGHT_EYE)
        ) / 2

        if ear < BLINK_THRESHOLD:
            blink_counter += 1
        else:
            if blink_counter >= BLINK_FRAMES and blink_cooldown == 0:
                filename = f"Captured/blink_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                blink_cooldown = 15
            blink_counter = 0

        if blink_cooldown > 0:
            blink_cooldown -= 1

    # ================= HAND DETECTION =================
    if hand_results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):

            handedness = hand_results.multi_handedness[i].classification[0].label
            state = hand_states[handedness]

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            fingers = []

            fingers.append(
                1 if hand_landmarks.landmark[4].x <
                     hand_landmarks.landmark[3].x else 0
            )

            for j in range(1, 5):
                fingers.append(
                    1 if hand_landmarks.landmark[TIP_IDS[j]].y <
                         hand_landmarks.landmark[TIP_IDS[j] - 2].y else 0
                )

            finger_count = fingers.count(1)
            state["finger_buffer"].append(finger_count)

            stable = None
            if len(state["finger_buffer"]) == 5:
                for fc in set(state["finger_buffer"]):
                    if state["finger_buffer"].count(fc) >= STABLE_THRESHOLD:
                        stable = fc
                        break

            if stable is not None and stable != state["last_finger"]:
                current_word += FINGER_TO_ASL.get(stable, "")
                state["last_finger"] = stable

    # ================= UI =================
    cv2.putText(
        frame,
        f"Word: {current_word}",
        (30, h - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        (0, 0, 255),
        3
    )

    cv2.imshow("ASL + Age Group + Blink Capture", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
