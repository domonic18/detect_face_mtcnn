# 构建自定义的脸部识别数据集
# 数据集使用CelebA数据集的图片和标签
import torch
import os
from torch.utils.data import DataLoader, Dataset 
from torchvision.transforms import ToTensor
from PIL import Image

class FaceDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.dataset = []
        self.trans = ToTensor()
        self.dataset.extend(open(os.path.join(self.path, "positive.txt")))
        self.dataset.extend(open(os.path.join(self.path, "negative.txt")))
        self.dataset.extend(open(os.path.join(self.path, "part.txt")))
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        # 从数据集列表中获取指定索引的样本字符串
        sample_str = self.dataset[index].strip().split(" ")
        
        # 解析样本信息
        img_filename = sample_str[0]
        cls = float(sample_str[1])
        offset = list(map(float, sample_str[2:]))
        
        # 构建图像路径
        img_path = os.path.join(self.path, img_filename)
        
        # 读取图像并应用变换
        img = Image.open(img_path)
        img = self.trans(img)
        
        # 返回图像、类别标签和偏移量
        return img, torch.tensor([cls]), torch.tensor(offset)