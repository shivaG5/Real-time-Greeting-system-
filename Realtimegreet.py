import cv2
import numpy as np
import pyttsx3

# Setup face recognizer and load model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Set label map
labels = {1: "Owner"}  # ensure your trainer used label '1' for owner

# Setup text-to-speech
engine = pyttsx3.init()
voices = engine.getProperty('voices')
for voice in voices:
    if "brian" in voice.name.lower() or ("english" in voice.name.lower() and "uk" in voice.name.lower()):
        engine.setProperty('voice', voice.id)
        break
engine.setProperty('rate', 150)

# Only greet once
greeted = False

# Start camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grayscale, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = grayscale[y:y+h, x:x+w]
        id_, confidence = recognizer.predict(roi_gray)

        if confidence < 90:  # more lenient
            name = labels.get(id_, "Owner")
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if not greeted:
                greeting = f"Hello {name}, welcome back!"
                print(greeting)
                engine.say(greeting)
                engine.runAndWait()
                greeted = True
        else:
            cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
