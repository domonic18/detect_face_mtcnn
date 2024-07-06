## 项目简介
这是一个使用MTCNN进行人脸识别的项目，该项目是结合开源的[MTCNN项目代码](https://www.jianshu.com/p/2227cc959aee)以及光辉课程中提供的代码，进行优化重构，一方面提升代码的可读性，另一方面未来可用于其他项目实战。

## MTCNN
MTCNN，Multi-task convolutional neural network（多任务卷积神经网络），将人脸区域检测与人脸关键点检测放在了一起，总体可分为P-Net、R-Net、和O-Net三层网络结构。

该模型主要采用了三个级联的网络，采用候选框加分类器的思想，进行快速高效的人脸检测。

## 运行环境
```python
matplotlib==3.8.4
numpy==2.0.0
opencv_python==4.8.0.74
Pillow==10.4.0
torch==2.3.1
torchvision==0.18.1
```
依赖环境已经在requirements.txt中列出，可以直接使用pip install -r requirements.txt进行安装。

## 目录结构
```shell
代码根目录
  |-datasets                                # 数据集目录
    |-celeba  
      |-identity_CelebA.txt
      |-list_attr_celeba.txt
      |-list_bbox_celeba.txt
      |-list_landmarks_align_celeba.txt
      |-list_landmarks_celeba.txt
      |-Img
        |-img_celeba
  |-doc                                     # 文档以及测试脚本
  |-model                                   # 模型文件
  |-train                                   # 训练脚本的库函数
  |-utils                                   # 工具函数
  |-gen_sample.py                           # 生成样本的脚本
  |-train_pnet.py                           # 训练P-Net的脚本
  |-train_rnet.py                           # 训练R-Net的脚本
  |-train_onet.py                           # 训练O-Net的脚本
```
## 运行步骤
1. **下载数据集**：首先需要下载数据集，数据集存放在datasets/celeba目录下，数据集下载地址为：[CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)。
2. **生成样本**：运行gen_sample.py脚本，生成样本。
3. **训练模型**：
   - P-Net：运行train_pnet.py脚本，训练P-Net。
   - R-Net：运行train_rnet.py脚本，训练R-Net。
   - O-Net：运行train_onet.py脚本，训练O-Net。
4. **预测模型**：(待补充代码)


## 参考资料
[CSDN：MTCNN之人脸检测——pytorch代码实现](https://www.jianshu.com/p/2227cc959aee)

[Gitee：Pytorch-MTCNN](https://gitee.com/yeyupiaoling/Pytorch-MTCNN?_from=gitee_search)

[Github：MTCNN-Tensorflow](https://github.com/AITTSMD/MTCNN-Tensorflow)