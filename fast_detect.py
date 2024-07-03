# import tool
from utils.tool import nms as NMS
from train import model_mtcnn as nets
import torch
import numpy as np
from torchvision import transforms
import time
from PIL import Image, ImageDraw
import cv2
import os
from torchvision.ops.boxes import batched_nms, nms

# 检测是否有GPU
device = "cuda:0" if torch.cuda.is_available() else 'cpu'


class Detector(object):
    def __init__(self,
                 pnet_path,
                 rnet_path,
                 onet_path,
                 softnms=False,
                 thresholds=(0.6, 0.6, 0.95),
                 factor=0.709):
        """
            初始化
        """
        # 三个网络的置信度阈值
        self.thresholds = thresholds

        # 缩放因子
        self.factor = factor

        # 是否启用softnms
        self.softnms = softnms

        # 构建模型
        self.pnet = nets.PNet().to(device)
        self.rnet = nets.RNet().to(device)
        self.onet = nets.ONet().to(device)

        # 加载参数
        self.pnet.load_state_dict(torch.load(pnet_path))
        self.rnet.load_state_dict(torch.load(rnet_path))
        self.onet.load_state_dict(torch.load(onet_path))

        # 设为评估模式
        self.pnet.eval()
        self.rnet.eval()
        self.onet.eval()

        # 跟训练时做相同的预处理
        self.img_transfrom = transforms.Compose([
            # 转张量 [0, 1]
            transforms.ToTensor(),
            # [0,1] --> [-1, 1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def detect(self, image):
        """
            主检测
        """
        start_time = time.time()

        # 第一步：传入P-Net做第一步的检测
        pnet_boxes = self.pnet_detect(image)

        print("pnet_boxes: ", pnet_boxes.shape)

        if pnet_boxes.shape[0] == 0:
            print("P网络未检测到人脸")
            return np.array([])

        end_time = time.time()
        pnet_time = end_time - start_time

        start_time = time.time()

        rnet_boxes = self.rnet_detect(image, pnet_boxes)
        print("rnet_boxes: ", rnet_boxes.shape)

        if rnet_boxes.shape[0] == 0:
            print("R网络未检测到人脸")
            return np.array([])

        end_time = time.time()
        rnet_time = end_time - start_time

        start_time = time.time()
        onet_boxes = self.onet_detect(image, rnet_boxes)
        print("onet_boxes: ", onet_boxes.shape)
        if onet_boxes.shape[0] == 0:
            print("O网路未检测到人脸")
            return np.array([])

        end_time = time.time()
        onet_time = end_time - start_time
        sum_time = pnet_time + rnet_time + onet_time
        print("time:{}, pnet_time:{}, rnet_time:{}, onet_time:{}".format(sum_time, pnet_time, rnet_time, onet_time))
        return pnet_boxes, rnet_boxes, onet_boxes

    def pnet_detect(self, image):
        """
            P-Net 检测

            - 传入的是 原始图像 的 Image对象

        """


        boxes = []

        print("P-Net 检测")
        # 遵从坐标习惯 H, W --> W, H

        w, h = image.size
        print("原始图像：", image.size)

        min_side = min(w, h)

        scale = 1
        scale_time = 1
        # 统计PNet检测的数量
        nums = 0
        # 图像金字塔
        while min_side > 12:
            # 图像预处理
            img_data = self.img_transfrom(image).to(device)
            # 添加批量维度 [3, h, w] --> [1, 3, h, w ]
            img_data.unsqueeze_(0)
            # print(f"第 {scale_time} 次检测 ")
            # print("输入图像：", img_data.shape)
            # 通过pnet模型，也就是正向传播
            with torch.no_grad():
                _cls, _offset = self.pnet(img_data)

            # [1, 1, H, W]
            print("输出置信度：", _cls.shape)
            print("输出bbox偏置：", _offset.shape)

            nums += _cls.shape[-1] * _cls.shape[-2]


            # 将数据搬到CPU上计算 [1, 1, 295, 445]
            _cls = _cls[0][0].data.cpu() # [295, 445]
            _offset = _offset[0].data.cpu() # [4, 295, 445]
            # [h, w]
            # print(_cls.shape)
            # print(_cls)
            # print(_offset.shape)
            # print(_offset)
            # (n,2) 阈值过滤 0.6
            # 返回每个满足条件的框的bbox的坐标

            indexes = torch.nonzero(_cls > self.thresholds[0])

            # 处理
            boxes.extend(self.box(indexes, _cls, _offset, scale))

            # 计算新的尺寸
            scale *= self.factor
            _w = int(w * scale)
            _h = int(h * scale)

            # 缩放图像（构建图像金字塔）
            image = image.resize((_w, _h))
            min_side = min(_w, _h)
            scale_time += 1
        # 打印PNet的检测次数
        print(nums)
        # 没有做去重复
        if self.softnms:
            return tool.soft_nms(torch.stack(boxes).numpy(), 0.3)


        # return tool.nms(torch.stack(boxes).numpy(), 0.3)
        boxes = torch.stack(boxes)
        return boxes[nms(boxes[:, :4], boxes[:, 4], 0.3)].numpy()

    def box(self, indexes, cls, offset, scale, stride=2, side_len=12):

        # 反向解码，映射到原始图像上
        # P-Net 反解坐标
        # 左上角坐标
        # anchor 的坐标是死的


        # 求映射到原图中的
        # 左上角的坐标
        _x1 = (indexes[:, 1] * stride) / scale
        _y1 = (indexes[:, 0] * stride) / scale

        # 右下角坐标
        _x2 = (indexes[:, 1] * stride + side_len) / scale
        _y2 = (indexes[:, 0] * stride + side_len) / scale

        # 边长
        side = _x2 - _x1

        # 取出有效框
        offset = offset[:, indexes[:, 0], indexes[:, 1]]

        # 通过误差和标准框，计算出真实预测框
        x1 = (_x1 + side * offset[0])
        y1 = (_y1 + side * offset[1])
        x2 = (_x2 + side * offset[2])
        y2 = (_y2 + side * offset[3])

        # (n,)
        cls = cls[indexes[:, 0], indexes[:, 1]]

        # (n, 5)
        result = torch.stack([x1, y1, x2, y2, cls], dim=1)
        return result

    def rnet_detect(self, image, pnet_boxes):
        """
            R-Net 检测
        """
        boxes = []
        img_dataset = []
        # 取出PNet的框，转为正方形，转成tensor，方便后面用tensor去索引
        square_boxes = torch.from_numpy(tool.convert_to_square(pnet_boxes))
        for box in square_boxes:
            _x1 = int(box[0])
            _y1 = int(box[1])
            _x2 = int(box[2])
            _y2 = int(box[3])
            # crop裁剪的时候超出原图大小的坐标会自动填充为黑色
            img_crop = image.crop([_x1, _y1, _x2, _y2])
            # 转为24*24，也就是RNet输入
            img_crop = img_crop.resize((24, 24))

            img_data = self.img_transfrom(img_crop).to(device)
            img_dataset.append(img_data)

        # (n,1) (n,4)
        _cls, _offset = self.rnet(torch.stack(img_dataset))

        _cls = _cls.data.cpu()
        _offset = _offset.data.cpu()

        # (14,)
        indexes = torch.nonzero(_cls > self.thresholds[1])[:, 0]

        # (n,5)
        box = square_boxes[indexes]

        # (n,)
        _x1 = box[:, 0]
        _y1 = box[:, 1]
        _x2 = box[:, 2]
        _y2 = box[:, 3]

        side = _x2 - _x1
        # (n,4)
        offset = _offset[indexes]
        # (n,)
        x1 = _x1 + side * offset[:, 0]
        y1 = _y1 + side * offset[:, 1]
        x2 = _x2 + side * offset[:, 2]
        y2 = _y2 + side * offset[:, 3]
        # (n,)
        cls = _cls[indexes][:, 0]

        # np.array([x1, y1, x2, y2, cls]) (5,n)
        boxes.extend(torch.stack([x1, y1, x2, y2, cls], dim=1))
        if len(boxes) == 0:
            return np.array([])

        boxes = torch.stack(boxes)
        return boxes[nms(boxes[:, :4], boxes[:, 4], 0.3)].numpy()

    def onet_detect(self, image, rnet_boxes):
        """
            O-Net 检测

        """
        boxes = []
        img_dataset = []
        square_boxes = tool.convert_to_square(rnet_boxes)
        for box in square_boxes:
            _x1 = int(box[0])
            _y1 = int(box[1])
            _x2 = int(box[2])
            _y2 = int(box[3])
            img_crop = image.crop([_x1, _y1, _x2, _y2])
            img_crop = img_crop.resize((48, 48))
            img_data = self.img_transfrom(img_crop).to(device)
            img_dataset.append(img_data)

        _cls, _offset, _point = self.onet(torch.stack(img_dataset))
        _cls = _cls.data.cpu().numpy()
        _offset = _offset.data.cpu().numpy()
        _point = _point.data.cpu().numpy()

        # 0.95
        indexes, _ = np.where(_cls > self.thresholds[2])

        # (n,5)
        box = square_boxes[indexes]

        # (n,)
        _x1 = box[:, 0]
        _y1 = box[:, 1]
        _x2 = box[:, 2]
        _y2 = box[:, 3]

        side = _x2 - _x1

        # (n,4)
        offset = _offset[indexes]

        # (n,)
        x1 = _x1 + side * offset[:, 0]
        y1 = _y1 + side * offset[:, 1]
        x2 = _x2 + side * offset[:, 2]
        y2 = _y2 + side * offset[:, 3]

        # (n,)
        cls = _cls[indexes][:, 0]
        # (n,10)
        point = _point[indexes]
        px1 = _x1 + side * point[:, 0]
        py1 = _y1 + side * point[:, 1]
        px2 = _x1 + side * point[:, 2]
        py2 = _y1 + side * point[:, 3]
        px3 = _x1 + side * point[:, 4]
        py3 = _y1 + side * point[:, 5]
        px4 = _x1 + side * point[:, 6]
        py4 = _y1 + side * point[:, 7]
        px5 = _x1 + side * point[:, 8]
        py5 = _y1 + side * point[:, 9]

        # np.array([x1, y1, x2, y2, cls, px1, py1, px2, py2, px3, py3, px4, py4, px5, py5]) (15,n)
        boxes.extend(np.stack([x1, y1, x2, y2, cls, px1, py1, px2, py2, px3, py3, px4, py4, px5, py5], axis=1))

        if len(boxes) == 0:
            return np.array([])

        # return tool.nms(np.stack(boxes), 0.3, isMin=True)
        return NMS(np.stack(boxes), 0.3, isMin=True)


if __name__ == '__main__':
    img_path = r"./datasets/test/detect_img/06.jpg"
    img = Image.open(img_path)
    current_path = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(current_path, "model")
    detector = Detector(os.path.join(base_path, "p_net.pt"), os.path.join(base_path, "r_net.pt"),
                        os.path.join(base_path, "o_net.pt"))
    # detector = Detector("model/p_net.pth", "model/r_net.pth", "model/o_net.pth")
    pnet_boxes, rnet_boxes, onet_boxes = detector.detect(img)
    img = cv2.imread(img_path)
    for box in onet_boxes:
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])
        cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=3)
        for i in range(5, 15, 2):
            cv2.circle(img, (int(box[i]), int(box[i + 1])), radius=2, color=(255, 255, 0), thickness=-1)
    cv2.imshow("img", img)
    cv2.waitKey(0)
