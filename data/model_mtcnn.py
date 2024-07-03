# 引入PyTorch
import torch
from torch import nn


class PNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 特征抽取
        self.features_extractor = nn.Sequential(
            # 第一层卷积
            nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(num_features=10),
            nn.ReLU(),

            # 第一层池化
            nn.MaxPool2d(kernel_size=3,stride=2, padding=1),

            # 第二层卷积
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),

            # 第三层卷积
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU()
        )

        # [N, 32, 1, 1] --> [N, 1, 1, 1]
        # 输出人脸的概率 bce 输出信息编码在了通道这个维度上
        self.cls_out = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0)
        
        # [N, 32, 1, 1] --> [N, 4, 1, 1]
        # 输出人脸的定位框的偏移量（误差）
        self.reg_out = nn.Conv2d(in_channels=32, out_channels=4, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # 抽取特征 [N, 32, 1, 1]
        x = self.features_extractor(x)
        # [N, 32, 1, 1] --> [N, 1, 1, 1]
        probs = torch.sigmoid(self.cls_out(x))
        offset = self.reg_out(x)

        return probs, offset


class RNet(nn.Module):
    def __init__(self):
        super(RNet, self).__init__()
        self.feature_extractor = nn.Sequential(
            # 第一层卷积 24 x 24
            nn.Conv2d(in_channels=3, out_channels=28, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(num_features=28),
            nn.ReLU(),

            # 第一层池化 11 x 11
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False),

            # 第二层卷积 9 x 9
            nn.Conv2d(in_channels=28, out_channels=48, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(num_features=48),
            nn.ReLU(),

            # 第二层池化 (没有补零) 4 x 4
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=False),

            # 第三层卷积 3 x 3
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),

            # 展平
            nn.Flatten(),

            # 全连接层 [batch_size, 128]
            nn.Linear(in_features=3 * 3 * 64, out_features=128)
        )


        # 类别
        self.cls_out = nn.Linear(in_features=128, out_features=1)
        # 误差回归
        self.reg_out = nn.Linear(in_features=128, out_features=4)

    def forward(self, x):
        x = self.feature_extractor(x)
        # 概率
        probs = torch.sigmoid(self.cls_out(x))
        # 误差
        offset = self.reg_out(x)

        return probs, offset


class ONet(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            # 第1层卷积 48 x 48
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),

            # 第1层池化 11 x 11
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False),

            # 第2层卷积 9 x 9
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),

            # 第2层池化 
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=False),

            # 第3层卷积 
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),

            # 第3层池化 
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False),

            # 第4层卷积
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),

            # 展平 [batch_size, n_features]
            nn.Flatten(),

            # 全连接 [batch_size, 128]
            nn.Linear(in_features=3 * 3 * 128, out_features=256)

        )

        # 概率输出
        self.cls_out = nn.Linear(in_features=256, out_features=1)

        # 回归量输出
        self.reg_out = nn.Linear(in_features=256, out_features=4)

        # 关键点输出
        self.landmark_out = nn.Linear(in_features=256, out_features=10)

    def forward(self, x):
        x = self.feature_extractor(x)
        probs = torch.sigmoid(self.cls_out(x))
        offset = self.reg_out(x)
        landmark = self.landmark_out(x)
        
        return probs, offset, landmark


if __name__ == '__main__':
    onet = ONet()
    print(onet)
    X = torch.randn(16, 3, 48, 48)
    print(X.shape)
    probs, offset, points = onet(X)
    print(probs.shape, offset.shape, points.shape)