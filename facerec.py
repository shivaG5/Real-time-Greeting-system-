import cv2
import face_recognition
import numpy as np
import pickle
from PIL import Image


pil_path = "C:/Users/SHIVA KUMAR/pro56/Opencv/Owner/Owner_fresh.jpg"
pil_image = Image.open(pil_path).convert("RGB")

safe_path = "C:/Users/SHIVA KUMAR/pro56/Opencv/Owner/safe_image.jpg"
pil_image.save(safe_path, format="JPEG")

bgr_image = cv2.imread(safe_path)
if bgr_image is None:
    raise FileNotFoundError(f"❌ Could not load sanitized image at {safe_path}")

print("✅ BGR Image loaded:", bgr_image.shape, bgr_image.dtype)


rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
clean_image = np.ascontiguousarray(rgb_image, dtype=np.uint8)

try:
    encodings = face_recognition.face_encodings(clean_image)
    if len(encodings) == 0:
        raise Exception("❌ No face found. Use a clear front-facing image.")
    
    with open("C:/Users/SHIVA KUMAR/pro56/Opencv/Owner_face.pkl", "wb") as f:
        pickle.dump(encodings[0], f)

    print("✅ Encoding extracted and saved successfully.")
except Exception as e:
    print("❌ face_recognition failed:", str(e))
