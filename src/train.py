import torch
from torch.utils.data import DataLoader
from data.FaceDataset import FaceDataset
import os


class Trainer:
    def __init__(self, net, param_path, data_path):
        # 检测是否有GPU
        self.device = 'cuda:0' if torch.cuda.is_available() else "cpu"
        # 把模型搬到device
        self.net = net.to(self.device)

        self.param_path = param_path

        # 打包数据
        self.datasets = FaceDataset(data_path)

        # 定义损失函数：类别判断（分类任务）
        self.cls_loss_func = torch.nn.BCELoss()

        # 定义损失函数：框的偏置回归
        self.offset_loss_func = torch.nn.MSELoss()

        # 定义损失函数：关键点的偏置回归
        self.point_loss_func = torch.nn.MSELoss()

        # 定义优化器
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=1e-3)

    def train(self, stop_value, landmark=False):
        """
            - 断点续传 --> 短点续训
            - transfer learning 迁移学习
            - pretrained model 预训练

        :param stop_value:
        :param landmark:
        :return:
        """

        # 加载上次训练的参数
        if os.path.exists(self.param_path):
            self.net.load_state_dict(torch.load(self.param_path))
            print("加载参数文件，继续训练 ...")
        else:
            print("没有参数文件，全新训练 ...")

        # 封装数据加载器
        dataloader = DataLoader(self.datasets, batch_size=32, shuffle=True)

        epochs = 0

        while True:
            # 训练一轮
            for i, (img_data, _cls, _offset, _point) in enumerate(dataloader):

                # 数据搬家 [32, 3, 12, 12]
                img_data = img_data.to(self.device)
                _cls = _cls.to(self.device)
                _offset = _offset.to(self.device)
                _point = _point.to(self.device)

                # O-Net输出三个
                if landmark:
                    out_cls, out_offset, out_point = self.net(img_data)
                else:
                    out_cls, out_offset = self.net(img_data)

                # [B, 1, 1, 1] --> [B, 1]
                out_cls = out_cls.view(-1, 1)

                # [B, 4, 1, 1] --> [B, 4]
                out_offset = out_offset.view(-1, 4)

                # 选取置信度为0，1的正负样本求置信度损失

                # 0: 负样本 1：正样本 2：偏样本
                # 提取出正样本和负样本的掩码
                cls_mask = torch.lt(_cls, 2)

                # 筛选复合条件的样本的：标签
                # 提取出正样本和负样本
                cls = torch.masked_select(_cls, cls_mask)

                # 筛选复合条件的样本的：预测值
                out_cls = torch.masked_select(out_cls, cls_mask)

                # 求解分类的 loss (正样本，负样本)
                cls_loss = self.cls_loss_func(out_cls, cls)

                # 选取正样本和部分样本求偏移率的损失

                # 选取正样本和偏样本
                # 提取正样本和偏样本的掩码
                offset_mask = torch.gt(_cls, 0)

                # bbox的标签
                offset = torch.masked_select(_offset, offset_mask)

                # bbox的预测值
                out_offset = torch.masked_select(out_offset, offset_mask)

                # bbox损失
                offset_loss = self.offset_loss_func(out_offset, offset)

                if landmark:

                    # 正和偏样本的landmark损失
                    point = torch.masked_select(_point, offset_mask)
                    out_point = torch.masked_select(out_point, offset_mask)
                    point_loss = self.point_loss_func(out_point, point)

                    # O-Net 的 最终损失！！！！！！！
                    loss = cls_loss + offset_loss + point_loss

                else:
                    # P-Net和R-Net 的 最终损失！！！！！！！
                    loss = cls_loss + offset_loss

                if landmark:
                    # dataloader.set_description(
                    #     "epochs:{}, loss:{:.4f}, cls_loss:{:.4f}, offset_loss:{:.4f}, point_loss:{:.4f}".format(
                    #         epochs, loss.float(), cls_loss.float(), offset_loss.float(), point_loss.float()))
                    print("loss:{0:.4f}, cls_loss:{1:.4f}, offset_loss:{2:.4f}, point_loss:{3:.4f}".format(
                        loss.float(), cls_loss.float(), offset_loss.float(), point_loss.float()))
                else:
                    # dataloader.set_description(
                    #     "epochs:{}, loss:{:.4f}, cls_loss:{:.4f}, offset_loss:{:.4f}".format(
                    #         epochs, loss.float(), cls_loss.float(), offset_loss.float()))
                    print("loss:{0:.4f}, cls_loss:{1:.4f}, offset_loss:{2:.4f}".format(
                        loss.float(), cls_loss.float(), offset_loss.float()))

                # 清空梯度
                self.optimizer.zero_grad()

                # 梯度回传
                loss.backward()

                # 优化
                self.optimizer.step()

            # 保存模型（参数）
            torch.save(self.net.state_dict(), self.param_path)

            # 轮次加1
            epochs += 1

            # 设定误差限制（此处有错误！！！用测试集上的平均损失，不能用训练集上的当前批次的损失）
            if loss < stop_value:
                break
