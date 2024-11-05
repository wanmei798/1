from time import time
import cv2
import torch
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from playsound import playsound  # 引入播放声音的库
ALARM_SOUND = '陶喆.mp3'

class ObjectDetection:
    # ... 省略了 __init__, predict, display_fps, plot_bboxes 方法的定义 ...

    def __call__(self):
        """Run object detection on video frames from a camera stream, plotting and showing the results."""
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened(), 'Failed to open video capture.'
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        frame_count = 0
        while True:
            self.start_time = time()
            ret, im0 = cap.read()
            assert ret, 'Failed to grab frame.'
            results = self.predict(im0)
            im0, class_ids = self.plot_bboxes(results, im0)

            # 播放声音文件代替发送邮件
            if len(class_ids) > 0:  # 只有在检测到对象时才播放声音
                if not self.email_sent:  # 假设 email_sent 用于控制声音播放
                    playsound(ALARM_SOUND)  # 播放声音文件
                    self.email_sent = True
            else:
                self.email_sent = False

            self.display_fps(im0)
            cv2.imshow("YOLOv8 Detection", im0)
            frame_count += 1
            if cv2.waitKey(5) & 0xFF == 27:  # 按 ESC 键退出
                break
        cap.release()
        cv2.destroyAllWindows()

# 创建 ObjectDetection 实例并运行
detector = ObjectDetection(capture_index=0)  # 传入摄像头索引
detector()
