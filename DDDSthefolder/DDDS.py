import cv2
from scipy.spatial import distance
import winsound

#Function calculate eye aspect ratio (EAR): 
#A and B are the vertical distances between the eye points
#C is the horizontal distance
#Idea: If the eyes are closed, A and B are small → EAR is small → indicates fatigue

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

#Loading the cascades:
#face_cascade detects faces and eye_cascade detects eyes within faces

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Opens the PC camera and checks that it is indeed open

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera not opened")
    exit()

#Fatigue detection parameters

EYE_AR_THRESH = 0.25        # If the eye ratio is less than this → considered closed
EYE_AR_CONSEC_FRAMES = 2   # how many consecutive frames need to be closed to trigger an alert
counter = 0                 # counter of frames with closed eyes

print("🎬 Starting Driver Drowsiness Detection...")

#Video processing loop:
#Reads each frame from the camera 
#Converts the image to grayscale because facial recognition works better on a gray image

while True:
    ret, frame = cap.read()
    if not ret:
        print("No frame captured")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # Detects the face in the frame, if a face is found → creates a blue square around each face

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)  # Detects eyes within the face, draws green squares around the eyes and calculates the aspect ratio (ear) for each eye
        eye_aspect_ratios = []

        if len(eyes) > 0:
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                # Standart calculation based on box (aspect ratio)
                ear = eh / float(ew)
                eye_aspect_ratios.append(ear)

            # Average of 2 eyes:
            # If eyes are closed (avg_ear < 0.25) → increment the counter, 
            # if the counter passes 15 consecutive frames → print an alert on the screen and also a sound (Beep),
            # if eyes are open → reset the counter
            avg_ear = sum(eye_aspect_ratios) / len(eye_aspect_ratios)
            if avg_ear < EYE_AR_THRESH:
                counter += 1
            else:
                counter = 0
        else:
            counter += 1

        if counter >= EYE_AR_CONSEC_FRAMES:
            cv2.putText(frame, "DROWSINESS ALERT!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            winsound.Beep(1000, 1000)
            counter = 0

    # Shows the video with the squares around the face and eyes, 
    # if eyes are closed for a long time → you see the text "DROWSINESS ALERT!" and you also hear the beep

    cv2.imshow("Driver Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Pressing q closes the window and the camera
        break

cap.release()
cv2.destroyAllWindows()
