import cv2
import os
import numpy as np

dataset_path = 'Dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []

# Loop through dataset images
for i, filename in enumerate(os.listdir(dataset_path)):
    if filename.endswith('.jpg'):
        img_path = os.path.join(dataset_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        faces.append(img)
        labels.append(0)  # Label 0 for "Owner"

# Train the model
recognizer.train(faces, np.array(labels))
recognizer.save("trainer.yml")
print("✅ Model trained and saved as 'trainer.yml'")
