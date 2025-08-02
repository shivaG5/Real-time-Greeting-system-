import face_recognition
import cv2
import pickle

bgr_image=cv2.imread("C://Users//SHIVA KUMAR//pro56//Opencv//Owner//Owner.jpg")
rgb_img=cv2.cvtColor(bgr_image,cv2.COLOR_BGR2RGB)
print("rgb_image shape:",rgb_img.shape)
print("RGB_image:",rgb_img)

encoding=face_recognition.face_encodings(rgb_img)

if len(encoding)==0:
    raise Exception("❌ No Face is found in the bgr_image.use clear photo")
encoding=encoding[0]
with open("Owner_face.pkl","wb") as f:
    pickle.dump(encoding,f)
print("Owner face encoding is saved")