import pyttsx3
import cv2 

engine = pyttsx3.init()

cam = cv2.VideoCapture(0)
while True:
    ret, frame = cam.read()
    if not ret:
        break

    rgb_frame = frame[:, :, ::-1]
    face_locations = Facedata.face_locations(rgb_frame)
    face_encodings = Facedata.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = Facedata.compare_faces(known_face, face_encoding)
        name = "Unknown"

        if True in matches:
            name = known_names[matches.index(True)]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, f"Hello, {name}!", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            engine.say(f"Hello {name}, welcome!")
            engine.runAndWait()
        else:
            cv2.putText(frame, "Unknown Face", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow("Live Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()
