from PIL import Image
from ultralytics import YOLO
import numpy as np


class PredictModel:
    def __init__(self):
        # 加载预训练的YOLOv8n模型
        self.model = YOLO('best-8643.pt')

    def __call__(self, img):
        # bgr
        results = self.model(img)
        assert len(results) == 1, "The detected image number must equal 1, but got {}".format(len(results))
        result = results[0]
        names, cls_info = self.post_proccessing(result)
        im_result = result.plot()  # 绘制包含预测结果的BGR numpy数组
        return results, im_result, names, cls_info

    def post_proccessing(self, result):
        names = result.names  # 检测的类别名称
        boxes = result.boxes  # 检测的框
        cls = boxes.cls.cpu().numpy().astype('int')  # 检测的类别索引
        keys, count = np.unique(cls, return_counts=True)  # 检测的每个类别的个数
        cls_info = {}
        # 整合检测的结果
        for key, c in zip(keys, count):
            cls_info[key] = c
        return names, cls_info
