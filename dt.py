import cv2
import os

# Load the Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Folders
image_folder = "Owner"
dataset_folder = "Dataset"
os.makedirs(dataset_folder, exist_ok=True)

# Loop through all 13 images
for i in range(1, 14):
    image_path = os.path.join(image_folder, f"img{i}.jpg")
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        print(f"❌ No face in {image_path}")
        continue

    for j, (x, y, w, h) in enumerate(faces):
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (200, 200))  # Resize to standard size
        filename = os.path.join(dataset_folder, f"user_{i}_{j}.jpg")
        cv2.imwrite(filename, face_resized)
        print(f"✅ Saved {filename}")
