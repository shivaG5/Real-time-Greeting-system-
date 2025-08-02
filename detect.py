import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from sms_alert import send_alert_sms
import time
from datetime import datetime
import numpy as np
model=YOLO("yolov8n.pt")
tracker=DeepSort(max_age=30)
cap=cv2.VideoCapture(0)
last_alert_time=0
alert_interval=30
recording=False
video_writer=None
record_start_time=0
record_duration=60

while True:
    ret,frame=cap.read()
    if not ret:
        break
    
    result=model.predict(frame, conf=0.4,iou=0.5)[0]
    detections=[]
    for r in result.boxes.data.tolist():
        x1,y1,x2,y2,score,class_id=r[:6]
        x1,y1,x2,y2=map(int,[x1,y1,x2,y2])
        class_name=model.names[int(class_id)]
        detections.append(([x1,y1,x2-x1,y2-y1,],float(score),class_name))
            
    tracks=tracker.update_tracks(detections,frame=frame)
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id=track.track_id
        l,t,w,h=track.to_ltrb()
        class_names=track.get_det_class() or "object"
        label=f"{class_names}|ID:{track_id}"
        cv2.rectangle(frame,(int(l),int(t)),(int (l+w),int(t+h)),(0,255,0),2)
        cv2.putText(frame,f'ID:{track_id}',(int(l), int(t-10)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
        
        if class_name=="person":
            if (time.time() -last_alert_time) > alert_interval:
             send_alert_sms("person deteceted on camera")
             last_alert_time=time.time()
             cv2.imwrite(f"intruder-{int(time.time())}.jpg",frame)
             
             #video_name=f"alert_{int(time.time())}.avi"
             time_stamp=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
             video_name=f"alert_{int(time.time())}.avi"
             fourcc=cv2.VideoWriter_fourcc(*'XVID')
             video_writer=cv2.VideoWriter(video_name,fourcc,20.0,(frame.shape[1],frame.shape[0]))
             recording=True
             record_start_time=time.time()
        
    if recording:
        video_writer.write(frame)
        if time.time()-record_start_time>record_duration:
            recording=False
            video_writer.release()
            print(f"video saved: {video_name}")
    
    cv2.imshow("YOLO + DEEPSORT",frame)
    if cv2.waitKey(1)& 0xff==ord('q'):
        break   
cap.release()
cv2.destroyAllWindows()
