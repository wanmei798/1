import sys
from PyQt5.QtWidgets import (QWidget, QPushButton,
                             QHBoxLayout, QVBoxLayout, QApplication, QLabel,
                             QFileDialog, QMessageBox, QRadioButton, QFrame)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer, Qt
import cv2
import os
from model import PredictModel


class MainGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.windows_title = "火焰烟雾识别"

        self.camera_stat = False  # 摄像头默认关闭
        self.cap = None  # 视频或者摄像头
        self.export_cap = None  # 导出的视频
        self.fname = None  # 图片或者视频路径

        self.init_ui()  # 初始化gui
        self.init_timer()  # 初始化定时器
        self.predictor = PredictModel()  # 模型预测的类

    def init_ui(self):
        h_layout = QHBoxLayout()  # 水平布局

        # 第一列布局，显示窗口  选择图片/视频按钮   打开摄像头按钮
        v1_layout = QVBoxLayout()

        v1_h1_layout = QHBoxLayout()
        self.display_label = QLabel()  # 用于显示图片的label
        self.display_label.setMinimumHeight(300)
        self.display_label.setMinimumWidth(400)
        self.display_label.setStyleSheet("border: 1px dashed")
        v1_h1_layout.addWidget(self.display_label)
        v1_layout.addLayout(v1_h1_layout)

        v1_h2_layout = QHBoxLayout()
        v1_h2_layout.addStretch(1)
        self.image_btn = QPushButton("选择图片/视频")  # 选择图片、视频按钮
        self.image_btn.clicked.connect(self.select_file)  # 选择图片视频按钮对应的事件
        v1_h2_layout.addWidget(self.image_btn)
        v1_h2_layout.addStretch(1)
        self.camera_btn = QPushButton("打开/关闭摄像头")  # 打开关闭摄像头按钮
        self.camera_btn.clicked.connect(self.open_camera)  # 打开关闭摄像头按钮对应的事件
        v1_h2_layout.addWidget(self.camera_btn)
        v1_h2_layout.addStretch(1)
        v1_layout.addLayout(v1_h2_layout)

        v1_frame = QFrame()
        # v1_frame.setStyleSheet("border: 1px dashed; border-color: (100, 100, 100);")
        v1_frame.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        v1_frame.setLayout(v1_layout)

        # 第二列 显示检测结果
        v2_layout = QVBoxLayout()

        dis_frame = QFrame()
        dis_frame.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        dis_frame.setFixedHeight(50)
        dis_layout = QHBoxLayout()
        self.display_detected_img_btn = QRadioButton("显示检测结果")  # 是否展示检测结果的勾选按钮
        self.display_detected_img_btn.setStyleSheet("border: 1px dashed")
        self.display_detected_img_btn.setChecked(True)
        dis_layout.addWidget(self.display_detected_img_btn)
        dis_frame.setLayout(dis_layout)
        v2_layout.addWidget(dis_frame)

        export_frame = QFrame()
        export_frame.setFixedHeight(120)
        export_frame.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        export_layout = QVBoxLayout()

        self.export_btn = QRadioButton("导出检测结果")  # 是否保存检测结果的选择按钮
        self.export_btn.setStyleSheet("border: 1px dashed")
        self.export_btn.setChecked(True)
        export_layout.addWidget(self.export_btn)

        self.export_path_btn = QPushButton("选择导出路径")  # 如果导出检测结果，选择对应的保存路径
        self.export_path_btn.setChecked(True)
        self.export_path_btn.clicked.connect(self.select_export_path)
        export_layout.addWidget(self.export_path_btn)

        default_export_path = os.path.join(os.getcwd(), 'outputs')
        os.makedirs(default_export_path, exist_ok=True)
        self.export_label = QLabel(default_export_path)
        self.export_label.setFixedHeight(20)
        self.export_label.setStyleSheet("border: 1px dashed; border-color: (100, 100, 100);")
        export_layout.addWidget(self.export_label)
        export_frame.setLayout(export_layout)
        v2_layout.addWidget(export_frame)

        self.info_label = QLabel()  # 展示最终的检测结果
        self.info_label.setStyleSheet("border: 1px dashed")
        v2_layout.addWidget(self.info_label)

        v2_frame = QFrame()
        # v2_frame.setStyleSheet("border: 1px dashed; border-color: (100, 100, 100);")
        v2_frame.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        v2_frame.setLayout(v2_layout)

        h_layout.addWidget(v1_frame)
        h_layout.addWidget(v2_frame)
        self.setLayout(h_layout)
        self.setWindowTitle(self.windows_title)
        # self.setGeometry(300, 300, 500, 500)
        self.show()

    def select_export_path(self):
        directory = QFileDialog.getExistingDirectory(None, "选择导出文件路径", "./")  # 选择导出的文件路径
        self.export_label.setText(directory)
        return directory

    def select_file(self):
        # 仅允许选择一个文件
        self.fname, _ = QFileDialog.getOpenFileName(self, '打开文件', "./",
                                                    "Files(*.jpg *.bmp *.png *.jpeg *.mp4 *.avi)")  # 选择文件仅允许图片和视频格式
        if self.fname == '':
            return
        if self.fname.endswith(('png', 'bmp', 'jpg', 'jpeg')):
            self.show_single_frame(self.fname)  # 如果是图片，处理图片即可
            self.fname = None  # 清空路径
        else:
            self.cap = cv2.VideoCapture(self.fname)  # 如果是视频，因为视频包含多个帧，逐帧处理，设定一个定时器，每间隔一定的时间就会处理一帧
            if not self.cap.isOpened():
                QMessageBox.information(self, "警告", "视频异常！", QMessageBox.Ok)
                self.cap = None
                return
            self.display_label.setEnabled(True)
            self.timer.start()  # 清空路径放在定时器
            print("beginning！")  #

    def open_camera(self):
        if self.camera_stat:  # 如果摄像头已经开启，那么再次点击就是关闭它
            self.camera_stat = not self.camera_stat
            self.close_camera()
            return
        self.cap = cv2.VideoCapture(1)
# 开启摄像头
        if not self.cap.isOpened():
            QMessageBox.information(self, "警告", "摄像头开启失败！", QMessageBox.Ok)
            self.cap = None
        else:
            self.camera_stat = True
            # 幕布可以播放
            self.display_label.setEnabled(True)  # 摄像头开启，开启定时器，逐帧播放
            self.timer.start()
            print("beginning！")

    def close_camera(self):
        # 关闭摄像头，关闭定时器，清空导出路径
        self.cap.release()
        self.cap = None
        self.timer.stop()
        if self.export_cap != None:  # 清空导出视频
            self.export_cap.release()
            self.export_cap = None
        self.fname = None

    # 播放视频画面
    def init_timer(self):
        # 初始化定时器，定时器的功能是按照间隔播放摄像头或者视频的内容
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.show_pic)

    # 显示视频图像
    def show_pic(self):
        # 如果是视频或者摄像头，逐帧检测结果
        ret, img = self.cap.read()
        if ret:
            self.show_single_frame(img)

    def show_single_frame(self, img):
        # 检测图片，或者视频和摄像头的一个帧
        image_path = None
        if isinstance(img, str):
            image_path = img
            img = cv2.imread(img)  # 如果是图片路径，则读取图片
        ##################################################det#####################################################
        result, det_img, names, cls_info = self.predictor(img)  # 对图片或者帧进行预测
        txt = ''  # 用于整合显示的检测结果
        print(names, cls_info)  # 打印相关信息
        for k in names.keys():
            name = names[k]
            if k in cls_info.keys():
                count = cls_info[k]
                txt += name + ': ' + str(count) + '\n'
            else:
                txt += name + ': ' + '0' + '\n'
        self.info_label.setText(txt)  # 将检测结果显示到gui的右下角框内
        if self.display_detected_img_btn.isChecked():  # 如果显示检测结果被选择，则显示检测结果
            img = det_img

        if self.export_btn.isChecked():  # 如果导出按钮被选择
            if image_path is not None:  # 检测的是图片
                image_name = os.path.basename(image_path)
                out_path = os.path.join(self.export_label.text(), image_name)
                cv2.imwrite(out_path, det_img)  # 保存检测结果
            else:  # 检测的是摄像头或者视频
                if self.export_cap is None:
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    if self.fname is not None:  # 如果是视频，保存其原有的名字，如果是摄像头，新建名字output.avi
                        video_name = os.path.basename(self.fname)
                        if 'mp4' in video_name:
                            video_name = video_name.replace('mp4', 'avi')
                            print(video_name)
                    else:
                        video_name = 'output.avi'
                    out_path = os.path.join(self.export_label.text(), video_name)
                    width = int(self.cap.get(3))
                    height = int(self.cap.get(4))
                    fps = self.cap.get(cv2.CAP_PROP_FPS)
                    self.export_cap = cv2.VideoWriter(out_path, fourcc, fps, (width, height), True)
                self.export_cap.write(det_img)  # 写入视频
        ##################################################det#####################################################
        cur_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 视频流的长和宽
        height, width = cur_frame.shape[:2]
        pixmap = QImage(cur_frame, width, height, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(pixmap)
        # 获取是视频流和label窗口的长宽比值的最大值，适应label窗口播放，不然显示不全
        ratio = max(width / self.display_label.width(), height / self.display_label.height())
        pixmap.setDevicePixelRatio(ratio)
        # 视频流置于label中间部分播放
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setPixmap(pixmap)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainGUI()
    sys.exit(app.exec_())
