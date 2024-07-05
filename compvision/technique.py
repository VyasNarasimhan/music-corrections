import cv2
from ultralytics import YOLO  
# define a video capture object
vid = cv2.VideoCapture(0)
bow_model = YOLO('./best (1).pt')
  
while vid.isOpened():

    success, frame = vid.read()

    if not success:
        break
    
    bow_detected = bow_model(frame, verbose=False, conf=0.9)
    for res in bow_detected:
        for box in res.boxes.xyxy:
            x1, y1, x2, y2 = box[:4]
            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2,
            )

    cv2.imshow('frame', frame)
      
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
vid.release()
cv2.destroyAllWindows()