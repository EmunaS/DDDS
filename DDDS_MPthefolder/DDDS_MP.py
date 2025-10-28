import cv2
import mediapipe as mp
import math
import winsound

#Basic settings
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
EYE_AR_THRESH = 0.25          #Eye-closing threshold
EYE_AR_CONSEC_FRAMES = 8     #How many consecutive frames will count as a blink
MOUTH_AR_THRESH = 0.6         #Open mouth = yawn
COUNTER_EYE = 0
COUNTER_MOUTH = 0

#Auxiliary function
def euclidean(p1, p2):
    return math.dist([p1.x, p1.y], [p2.x, p2.y])

def eye_aspect_ratio(landmarks, eye_indices):
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye_indices]
    A = euclidean(p2, p6)
    B = euclidean(p3, p5)
    C = euclidean(p1, p4)
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(landmarks, mouth_indices):
    p1, p2, p3, p4, p5, p6, p7, p8 = [landmarks[i] for i in mouth_indices]
    A = euclidean(p2, p8)
    B = euclidean(p3, p7)
    C = euclidean(p4, p6)
    D = euclidean(p1, p5)
    return (A + B + C) / (3.0 * D)

#Index of points in face (MediaPipe)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 81, 13, 14, 311, 308, 402, 318]

#Camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            #Calculating EAR for the eyes
            leftEAR = eye_aspect_ratio(landmarks, LEFT_EYE)
            rightEAR = eye_aspect_ratio(landmarks, RIGHT_EYE)
            ear = (leftEAR + rightEAR) / 2.0

            #Calculation MAR for the mouth
            mar = mouth_aspect_ratio(landmarks, MOUTH)

            #Drawing data on the screen
            cv2.putText(frame, f"EAR: {ear:.2f}", (10, 430),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (10, 460),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            #Fatigue detection
            if ear < EYE_AR_THRESH:
                COUNTER_EYE += 1
                if COUNTER_EYE >= EYE_AR_CONSEC_FRAMES:
                    cv2.putText(frame, "DROWSINESS ALERT!", (120, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    winsound.Beep(1000, 500)
            else:
                COUNTER_EYE = 0

            #Yawn detection
            if mar > MOUTH_AR_THRESH:
                COUNTER_MOUTH += 1
                if COUNTER_MOUTH >= 10:
                    cv2.putText(frame, "YAWNING ALERT!", (150, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    winsound.Beep(800, 500)
            else:
                COUNTER_MOUTH = 0

    cv2.imshow("Driver Drowsiness & Yawn Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
