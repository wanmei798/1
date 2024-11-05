from ultralytics import YOLO
import cv2
model = YOLO('best-8643.pt')

cap = cv2.VideoCapture(1) # 0表示调用摄像头

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)
    annotated_frame = results[0].plot()
    cv2.imshow('YOLOv8', annotated_frame)
    if cv2.waitKey(1) == ord('q'):  # 按下q退出
        break

cap.release()
cv2.destroyAllWindows()

