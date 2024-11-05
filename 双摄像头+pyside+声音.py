import sys
import cv2
from PySide6.QtWidgets import (QWidget, QPushButton,
                               QHBoxLayout, QVBoxLayout, QApplication, QLabel)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import QTimer, Qt
from ultralytics import YOLO
import numpy as np
from threading import Thread
from playsound import playsound  # 引入播放声音的库
ALARM_SOUND = '3秒mp3.MP3'

class CameraWidget(QLabel):
    def __init__(self, device, model, parent=None):
        super().__init__(parent)
        self.device = device
        self.model = model
        self.cap = cv2.VideoCapture(device)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.is_active = False  # 新增的状态变量，用于控制是否激活
        self.setFixedSize(400, 300)  # 固定大小
        self.setAlignment(Qt.AlignCenter)  # 居中文本
        self.setStyleSheet("background-color: lightGray;")  # 背景颜色
        self.setText("请打开摄像头")

    def update_frame(self):
        if self.is_active:
            ret, frame = self.cap.read()
            if ret:
                results = self.model.track(frame, persist=True,conf=0.5)
                for r in results:
                    boxes = r.boxes  # Boxes object for bbox outputs
                    class_ids = boxes.cls.cpu().numpy().astype(int)  # 转为int类型数组
                    num1 = np.sum(class_ids == 0)  # 统计类别为0的检测框数目
                    num2 = np.sum(class_ids == 1)  # 统计类别为0的检测框数目
                    num = num1 + num2
                    if num > 0:  # 如果检测到物体
                        # 在新线程中播放报警声
                        Thread(target=playsound, args=(ALARM_SOUND,), daemon=True).start()

                res_plotted = results[0].plot()
                rgb_image = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(convert_to_Qt_format)
                self.setPixmap(pixmap.scaled(300, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def start_capture(self):
        self.is_active = True
        self.timer.start(30)  # 更新频率约为33fps
        self.setText("")  # 清空文本

    def stop_capture(self):
        self.is_active = False
        self.timer.stop()
        self.setText("请打开摄像头")  # 显示提示信息

    def closeEvent(self, event):
        self.stop_capture()
        self.cap.release()
        super().closeEvent(event)

class MainGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        # 加载YOLO模型
        self.model1 = YOLO('best-1030.pt')
        self.model2 = YOLO('best-1030.pt')

        h_layout = QHBoxLayout()  # 主水平布局

        # 第一行：两个视频画面显示区域
        top_v_layout = QVBoxLayout()  # 上方的垂直布局

        # 创建两个摄像头小部件
        self.camera0 = CameraWidget(0, self.model1, self)
        self.camera1 = CameraWidget(1, self.model2, self)

        # 将摄像头小部件添加到垂直布局中
        top_v_layout.addWidget(self.camera0)
        top_v_layout.addWidget(self.camera1)

        # 第二行：两个按钮
        bottom_h_layout = QHBoxLayout()  # 下方的水平布局

        # 创建两个按钮
        self.button0 = QPushButton("打开摄像头0", self)
        self.button0.clicked.connect(lambda: self.toggle_camera(0))
        self.button1 = QPushButton("打开摄像头1", self)
        self.button1.clicked.connect(lambda: self.toggle_camera(1))

        # 将按钮添加到水平布局中
        bottom_h_layout.addWidget(self.button0)
        bottom_h_layout.addWidget(self.button1)

        # 将上下布局合并到主布局中
        h_layout.addLayout(top_v_layout)
        h_layout.addLayout(bottom_h_layout)

        # 设置窗口布局
        self.setLayout(h_layout)
        self.setWindowTitle('双摄像头显示')
        self.setGeometry(100, 100, 800, 600)
        self.show()

    def toggle_camera(self, device):
        if device == 0:
            if self.camera0.is_active:
                self.camera0.stop_capture()
                self.button0.setText("打开摄像头0")
            else:
                self.camera0.start_capture()
                self.button0.setText("关闭摄像头0")
        elif device == 1:
            if self.camera1.is_active:
                self.camera1.stop_capture()
                self.button1.setText("打开摄像头1")
            else:
                self.camera1.start_capture()
                self.button1.setText("关闭摄像头1")

if __name__ == '__main__':
    app = QApplication(sys.argv)

    ex = MainGUI()
    sys.exit(app.exec())
