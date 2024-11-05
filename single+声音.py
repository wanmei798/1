import cv2
import numpy as np
from ultralytics import YOLO
from playsound import playsound  # 引入播放声音的库

# 确保文件路径正确
MODEL_PATH = 'best-8643.pt'
ALARM_SOUND = '3秒mp3.MP3'

# 加载 YOLOv8 模型
model = YOLO(MODEL_PATH)

# 获取摄像头内容，参数 1 表示使用第二个摄像头
cap1 = cv2.VideoCapture(1)

while cap1.isOpened():
    success, frame = cap1.read()  # 读取摄像头的一帧图像
    if not success:
        print("无法获取帧")
        break

    # 对当前帧进行目标检测并显示结果
    results = model(frame, stream=True, show=True)

    for r in results:
        boxes = r.boxes  # Boxes object for bbox outputs
        class_ids = boxes.cls.cpu().numpy().astype(int)  # 转为int类型数组
        num = np.sum(class_ids == 0)  # 统计类别为0的检测框数目

        if num > 0:  # 如果检测到物体
            playsound(ALARM_SOUND)  # 播放报警声

cap1.release()  # 释放摄像头资源
cv2.destroyAllWindows()  # 关闭OpenCV窗口
