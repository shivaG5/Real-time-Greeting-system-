import face_recognition
import cv2 as cv
import os
known_face=[]
known_names=[]
dataset_path="dataset"
for filename in os.listdir(dataset_path):
    image_path=os.path.join(dataset_path,filename)
    image=face_recognition.load_image_file(image_path)
    try:
        encoding=face_recognition.face_encodings(image)[0]
        known_face.append(encoding)
        known_names("Shiva")
    except:
        print(f"Face is not detected in{filename}")
        
    