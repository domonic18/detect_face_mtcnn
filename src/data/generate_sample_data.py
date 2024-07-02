"""
模块功能：celeb A数据集转换成12, 24, 48大小的训练数据
"""

import os
import random
import numpy as np
import torch
from PIL import Image
from utils import IOU

# 文件路径配置
"""
datasets/
    |-celeba/
        |-img_align_celeba/
            |-000001.jpg
            |-000002.jpg
            ...
        |-list_bbox_celeba.txt
    |-train/
"""
current_path = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.join(current_path, "../../datasets")
TARGET_PATH = os.path.join(BASE_PATH, "celeba")
IMG_PATH = os.path.join(BASE_PATH, "celeba/img_celeba")
DST_PATH = os.path.join(BASE_PATH, "train")
LABEL_PATH = os.path.join(TARGET_PATH, "list_bbox_celeba.txt")
LANMARKS_PATH = os.path.join(TARGET_PATH, "list_landmarks_celeba.txt")

# 测试样本个数限制,设置为 -1 表示全部
TEST_SAMPLE_LIMIT = 100

def create_directories(base_path, face_size):
    paths = {}
    base_path = os.path.join(base_path, f"{face_size}")
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    
    paths['positive'] = os.path.join(base_path, "positive")
    paths['negative'] = os.path.join(base_path, "negative")
    paths['part'] = os.path.join(base_path, "part")
    
    for path in paths.values():
        if not os.path.exists(path):
            os.makedirs(path)
    
    return paths, base_path

def open_label_files(base_path):
    files = {}
    files['positive'] = open(os.path.join(base_path, "positive.txt"), "w")
    files['negative'] = open(os.path.join(base_path, "negative.txt"), "w")
    files['part'] = open(os.path.join(base_path, "part.txt"), "w")
    return files

def parse_annotation_line(line):
    strs = line.strip().split(" ")
    strs = list(filter(bool, strs))
    return strs

def adjust_bbox(x1, y1, w, h):
    # 标注不标准，给框适当的偏移量
    x1 = int(x1 + w * 0.12)
    y1 = int(y1 + h * 0.1)
    
    x2 = int(x1 + w * 0.9)
    y2 = int(y1 + h * 0.85)

    w = int(x2 - x1)
    h = int(y2 - y1)
    
    return x1, y1, x2, y2, w, h

def generate_crop_boxes(cx, cy, w, h):
    """
    根据给定的人脸中心点坐标和尺寸,生成5个候选的裁剪框。
    
    参数:
    cx (float): 人脸中心点的 x 坐标
    cy (float): 人脸中心点的 y 坐标
    w (int): 人脸框的宽度
    h (int): 人脸框的高度
    
    返回:
    crop_boxes (list): 一个包含5个裁剪框坐标的列表,每个裁剪框的格式为 [x1, y1, x2, y2]
    """
    crop_boxes = []
    for _ in range(5):
        # 生成随机的偏移量,限制在人脸框宽高的20%范围内
        w_offset = np.random.uniform(max(-w * 0.2, -w), min(w * 0.2, w))
        h_offset = np.random.uniform(max(-h * 0.2, -h), min(h * 0.2, h))
        
        # 计算新的中心点坐标,并限制在图像边界内
        # cx_ = np.clip(cx + w_offset, 0, w)
        # cy_ = np.clip(cy + h_offset, 0, h)
        cx_ = cx + w_offset
        cy_ = cy + h_offset
        
        # 随机生成边长,范围为原人脸框宽高的80%到125%
        side_len = np.random.uniform(min(w, h) * 0.8, max(w, h) * 1.25)
        side_len = round(side_len)
        
        # 计算裁剪框的坐标,确保在图像边界内
        x1_ = np.max([cx_ - side_len // 2, 0])
        y1_ = np.max([cy_ - side_len // 2, 0])
        # x2_ = np.min([x1_ + side_len, w])
        # y2_ = np.min([y1_ + side_len, h])
        x2_ = x1_ + side_len
        y2_ = y1_ + side_len
        
        # 添加裁剪框坐标到列表中
        crop_boxes.append(np.array([x1_, y1_, x2_, y2_]))
    
    return crop_boxes

def process_crop_box(img, crop_box, boxes, face_size, 
                     base_path, positive_counter, negative_counter, part_counter, 
                     positive_file, negative_file, part_file):
    # 确保只取前四个元素
    x1, y1, x2, y2 = boxes[0][:4]  
    side_len = crop_box[2] - crop_box[0]
    offset_x1 = (x1 - crop_box[0]) / side_len
    offset_y1 = (y1 - crop_box[1]) / side_len
    offset_x2 = (x2 - crop_box[2]) / side_len
    offset_y2 = (y2 - crop_box[3]) / side_len

    face_crop = img.crop(crop_box)
    face_resize = face_crop.resize((face_size, face_size), Image.Resampling.LANCZOS)
    
    iou = IOU(torch.tensor([x1, y1, x2, y2]), torch.tensor([crop_box[:4]]))


    if iou > 0.6: # 正样本
        face_resize.save(os.path.join(base_path, f"positive/{positive_counter}.jpg"))
        positive_file.write(f"positive/{positive_counter}.jpg 1 {offset_x1} {offset_y1} {offset_x2} {offset_y2}\n")
        positive_counter += 1
    elif iou > 0.4: # 部分样本
        face_resize.save(os.path.join(base_path, f"part/{part_counter}.jpg"))
        part_file.write(f"part/{part_counter}.jpg 2 {offset_x1} {offset_y1} {offset_x2} {offset_y2}\n")
        part_counter += 1
    elif iou < 0.1: # 负样本
        face_resize.save(os.path.join(base_path, f"negative/{negative_counter}.jpg"))
        negative_file.write(f"negative/{negative_counter}.jpg 0 0 0 0 0\n")
        negative_counter += 1

    # 随机采样负样本：为了提升负样本的数量
    random_x1 = random.randint(0, img.width - face_size)
    random_y1 = random.randint(0, img.height - face_size)
    random_x2 = random_x1 + face_size
    random_y2 = random_y1 + face_size
    random_box = [random_x1, random_y1, random_x2, random_y2]

     # 计算随机框与人脸框的IoU,如果小于阈值则保存为负样本
    iou_random = IOU(torch.tensor([random_x1, random_y1, random_x2, random_y2]), torch.tensor(boxes))
    if torch.max(iou_random) < 0.15:
        face_resize = img.crop([random_x1, random_y1, random_x2, random_y2]).resize((face_size, face_size), Image.Resampling.LANCZOS)
        face_resize.save(os.path.join(base_path, f"negative/{negative_counter}.jpg"))
        negative_file.write(f"negative/{negative_counter}.jpg 0 0 0 0 0\n")
        negative_counter += 1

    return positive_counter, negative_counter, part_counter

def process_annotation_line(line, face_size, positive_counter, negative_counter, part_counter, 
                            positive_file, negative_file, part_file, 
                            base_path, img_path):
    """
    处理一行注释信息,生成正负样本并保存到文件中。
    
    参数:
    line (str): 一行注释信息,格式为 "image_filename x1 y1 w h"
    face_size (int): 生成的人脸图像尺寸
    positive_counter (int): 正样本计数器
    negative_counter (int): 负样本计数器
    part_counter (int): 部分样本计数器
    positive_file (file): 正样本输出文件
    negative_file (file): 负样本输出文件
    part_file (file): 部分样本输出文件
    base_path (str): 输出文件的基础路径
    img_path (str): 输入图像的路径
    
    返回:
    positive_counter (int), negative_counter (int), part_counter (int): 更新后的计数器值
    """
    # 解析注释行,获取图像文件名和人脸位置信息
    strs = parse_annotation_line(line)
    image_filename = strs[0]
    x1, y1, w, h = map(int, strs[1:])
    
    # 调整人脸框的坐标
    x1, y1, x2, y2, w, h = adjust_bbox(x1, y1, w, h)
    boxes = [[x1, y1, x2, y2]]
    
    # 计算人脸中心点坐标
    cx = w / 2 + x1
    cy = h / 2 + y1

    # 打开图像文件
    image_filepath = os.path.join(img_path, image_filename)
    with Image.open(image_filepath) as img:
        # 生成候选的裁剪框
        for crop_box in generate_crop_boxes(cx, cy, w, h):
            # 处理每个候选裁剪框,保存正负样本
            positive_counter, negative_counter, part_counter = process_crop_box(
                img, crop_box, boxes, face_size, base_path, positive_counter, negative_counter, part_counter,
                positive_file, negative_file, part_file
            )
    
    # 返回更新后的计数器值
    return positive_counter, negative_counter, part_counter

def gen_sample(face_size, stop_value=100):
    """
        face_size: 图像大小，12， 24， 48
        stop_value：图像总的个数
    """
    if not os.path.exists(DST_PATH):
        os.makedirs(DST_PATH)

    paths, base_path = create_directories(DST_PATH, face_size)
    files = open_label_files(base_path)

    positive_counter = 0
    negative_counter = 0
    part_counter = 0

    for i, line in enumerate(open(label_file_path)):
        print(f"positive:{positive_counter}, negative:{negative_counter}, part:{part_counter}")
        # 跳过前两行
        if i < 2:
            continue
        
        # 如果处理了指定数量的样本,则退出循环
        if stop_value > 0 and i > stop_value:
            break
        
        # 按行读取5个关键点的标签文件，返回一个列表【关键点】
        with open(LANMARKS_PATH) as f:
            landmarks_list = f.readlines()

        # 读取CelebA的标签文件【框的信息】
        with open(LABEL_PATH) as f:
            anno_list = f.readlines()

        positive_counter, negative_counter, part_counter = process_annotation_line(
            line, face_size, positive_counter, negative_counter, part_counter,
            files['positive'], files['negative'], files['part'], base_path, IMG_PATH
        )

    for file in files.values():
        file.close()


def main():
    # 生成12×12的样本
    gen_sample(12, 100)

    gen_sample(24, 100)

    gen_sample(48, 100)

if __name__ == "__main__":
    main()
