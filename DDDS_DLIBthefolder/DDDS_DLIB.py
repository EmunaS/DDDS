import cv2
import dlib
from scipy.spatial import distance
import imutils

#Function to calculate aspect ratio of the eye(Eye Aspect Ratio)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

#Thresholds
EYE_AR_THRESH = 0.25   #Below that, the eye is considered closed
EYE_AR_CONSEC_FRAMES = 20  #After a few frames with eyes closed, a warning will be triggered

COUNTER = 0

#Initialize face and landmarks detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:\PythonFolder\DriverDrowsinessDetectionSystem\shape_predictor_68_face_landmarks.dat")

#Eye point identifiers
(lStart, lEnd) = (42, 48)
(rStart, rEnd) = (36, 42)

#Camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        coords = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

        leftEye = coords[lStart:lEnd]
        rightEye = coords[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        #Drawing the eyes
        for (x, y) in leftEye + rightEye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        #If eyes are closed long enough → warning
        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame, "DROWSINESS ALERT!", (150, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                #Simple beep
                import winsound
                winsound.Beep(1000, 500)
        else:
            COUNTER = 0

        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Driver Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
