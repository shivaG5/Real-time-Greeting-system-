import cv2 as cv
import os
cam=cv.VideoCapture(0)
cv.namedWindow("Myself")
count=0
os.makedirs("dataset",exist_ok=True)
while True:
    ret,frame=cam.read()
    if not ret:
        break
    cv.imshow("Face is Capturing",frame)
    k=cv.waitKey(1)
    if k%256==32:
        img_name=f"dataset/face_{count}.jpg"
        cv.imwrite(img_name,frame)
        print(f"✅ saved{img_name}")
        count+=1
    elif k%256==27:
        break
    elif k==ord("q"):
        break
    
cam.release()
cv.destroyAllWindows()
    
