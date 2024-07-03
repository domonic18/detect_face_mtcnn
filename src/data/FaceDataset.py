# 构建自定义的脸部识别数据集
# 数据集使用CelebA数据集的图片和标签
import torch
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset 
from PIL import Image

class FaceDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.dataset = []
        self._read_annos()

    def _read_annos(self):
        with open(os.path.join(self.path, "positive.txt")) as f:
            self.datasets.extend(f.readlines())

        with open(os.path.join(self.path, "negative.txt")) as f:
            self.datasets.extend(f.readlines())

        with open(os.path.join(self.path, "part.txt")) as f:
            self.datasets.extend(f.readlines())

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        strs = self.datasets[idx].strip().split()
        # 文件名字
        img_name = strs[0]

        # 取出类别
        cls = torch.tensor([int(strs[1])], dtype=torch.float32)

        # 将所有偏置转为float类型
        strs[2:] = [float(x) for x in strs[2:]]

        # bbox的偏置
        offset = torch.tensor(strs[2:6], dtype=torch.float32)
        # landmark的偏置
        point = torch.tensor(strs[6:16], dtype=torch.float32)

        # 打开图像
        img = Image.open(os.path.join(self.path, img_name))

        # 数据调整到 [-1, 1]之间
        img_data = torch.tensor((np.array(img) / 255. - 0.5) / 0.5, dtype=torch.float32)
        # [H, W, C] --> [C, H ,W]
        img_data = img_data.permute(2, 0, 1)

        return img_data, cls, offset, point