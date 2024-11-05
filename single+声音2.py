import cv2
import numpy as np
from ultralytics import YOLO
from threading import Thread
from playsound import playsound  # 引入播放声音的库

MODEL_PATH = 'best-20240903.pt'
ALARM_SOUND = '警报(1).MP3'

# 加载 YOLOv8 模型
model = YOLO(MODEL_PATH)

# 获取摄像头内容，参数 1 表示使用第二个摄像头
cap1 = cv2.VideoCapture(1)

def play_sound(sound_file):
    """在一个新线程中播放声音文件"""
    playsound(sound_file)

while cap1.isOpened():
    success, frame = cap1.read()  # 读取摄像头的一帧图像
    if not success:
        print("无法获取帧")
        break

    # 对当前帧进行目标检测
    results = model(frame, stream=True, show=True)

    for r in results:
        boxes = r.boxes  # Boxes object for bbox outputs
        class_ids = boxes.cls.cpu().numpy().astype(int)  # 转为int类型数组
        num1 = np.sum(class_ids == 0)  # 统计类别为0的检测框数目
        num2 = np.sum(class_ids == 1)  # 统计类别为0的检测框数目
        num = num1+ num2
        if num > 0:  # 如果检测到物体
            # 在新线程中播放报警声
            Thread(target=play_sound, args=(ALARM_SOUND,), daemon=True).start()
    # 按'q'键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()  # 释放摄像头资源
cv2.destroyAllWindows()  # 关闭OpenCV窗口
