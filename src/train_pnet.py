from data import model_mtcnn as nets
import os
import train


if __name__ == '__main__':
    
    current_path = os.path.dirname(os.path.abspath(__file__))
    # 权重存放地址
    base_path = os.path.join(current_path, "../model")
    model_path = os.path.join(base_path, "p_net.pt")

    # 数据存放地址
    data_path = os.path.join(current_path, "../datasets/train/12")
    
    # 如果没有这个参数存放目录，则创建一个目录
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    # 构建模型
    pnet = nets.PNet()

    # 开始训练
    t = train.Trainer(pnet, model_path, data_path)

    t.train(0.01)
