"""
模块功能：celeb A数据集转换成12, 24, 48大小的训练数据
"""

import os
import random
import numpy as np
import torch
from PIL import Image
from utils.tool import iou as IOU

# 文件路径配置
"""
datasets/
    |-celeba/
        |-img_celeba/
            |-000001.jpg
            |-000002.jpg
            ...
        |-list_bbox_celeba.txt
        |-list_landmarks_celeba.txt
    |-train/
"""
current_path = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.join(current_path, "datasets")
TARGET_PATH = os.path.join(BASE_PATH, "celeba")
IMG_PATH = os.path.join(BASE_PATH, "celeba/img_celeba")
DST_PATH = os.path.join(BASE_PATH, "train")
LABEL_PATH = os.path.join(TARGET_PATH, "list_bbox_celeba.txt")
LANMARKS_PATH = os.path.join(TARGET_PATH, "list_landmarks_celeba.txt")

# 测试样本个数限制,设置为 -1 表示全部
TEST_SAMPLE_LIMIT = 100

# 为随机数种子做准备，使正样本，部分样本，负样本的比例为1：1：3
float_num = [0.1, 0.1, 0.3, 0.5, 0.95, 0.95, 0.99, 0.99, 0.99, 0.99]

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
    strs = line.strip().split()
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

def generate_crop_boxes(cx, cy, max_side, img_w, img_h):
    """
    根据给定的人脸中心点坐标和尺寸,生成5个候选的裁剪框。
    
    参数:
    cx (float): 人脸中心点的 x 坐标
    cy (float): 人脸中心点的 y 坐标
    max_side (int): 人脸框的最大边长
    img_w (int): 图像宽度
    img_h (int): 图像高度
    
    返回:
    crop_boxes (list): 一个包含5个裁剪框坐标的列表,每个裁剪框的格式为 [x1, y1, x2, y2]
    """

    crop_boxes = []
    for _ in range(5):
        # 随机偏移中心点坐标以及边长
        seed = float_num[np.random.randint(0, len(float_num))]

        # 最大边长随机偏移
        _max_side = max_side + np.random.randint(int(-max_side * seed), int(max_side * seed))

        # 中心点x坐标随机偏移
        _cx = cx + np.random.randint(int(-cx * seed), int(cx * seed))

        # 中心点y坐标随机偏移
        _cy = cy + np.random.randint(int(-cy * seed), int(cy * seed))

        # 得到偏移后的坐标值（方框）
        _x1 = _cx - _max_side / 2
        _y1 = _cy - _max_side / 2
        _x2 = _x1 + _max_side
        _y2 = _y1 + _max_side

        # 偏移过大，偏出图像了，此时，不能用，应该再次尝试偏移
        if _x1 < 0 or _y1 < 0 or _x2 > img_w or _y2 > img_h:
            continue
        
        # 添加裁剪框坐标到列表中
        crop_boxes.append(np.array([_x1, _y1, _x2, _y2]))
    
    return crop_boxes

def process_crop_box(img, face_size, max_side, crop_box, boxes, landmarks):
    """
    处理单个裁剪框,生成正负样本。
    
    参数:
    img (Image): 原始图像
    crop_box (list): 裁剪框坐标 [x1, y1, x2, y2]
    boxes (list): 人脸框坐标列表
    face_size (int): 生成的人脸图像尺寸
    
    返回:
    sample (dict): 样本信息 {'image': image, 'label': label, 'bbox_offsets': offsets, 'landmark_offsets': landmark_offsets}
    """
    x1, y1, x2, y2 = boxes[0][:4]
    _x1, _y1, _x2, _y2 = crop_box[:4]
    px1, py1, px2, py2, px3, py3, px4, py4, px5, py5 = landmarks
    _max_side = max_side


    offset_x1 = (x1 - _x1) / _max_side
    offset_y1 = (y1 - _y1) / _max_side
    offset_x2 = (x2 - _x2) / _max_side
    offset_y2 = (y2 - _y2) / _max_side

    offset_px1 = (px1 - _x1) / _max_side
    offset_py1 = (py1 - _y1) / _max_side
    offset_px2 = (px2 - _x1) / _max_side
    offset_py2 = (py2 - _y1) / _max_side
    offset_px3 = (px3 - _x1) / _max_side
    offset_py3 = (py3 - _y1) / _max_side
    offset_px4 = (px4 - _x1) / _max_side
    offset_py4 = (py4 - _y1) / _max_side
    offset_px5 = (px5 - _x1) / _max_side
    offset_py5 = (py5 - _y1) / _max_side



    face_crop = img.crop(crop_box)
    face_resize = face_crop.resize((face_size, face_size), Image.Resampling.LANCZOS)
    
    iou = IOU(torch.tensor([x1, y1, x2, y2]), torch.tensor([crop_box[:4]]))

    if iou > 0.7:  # 正样本
        label = 1
    elif 0.4 < iou < 0.6:  # 部分样本
        label = 2
    elif iou < 0.2:  # 负样本
        label = 0
    else:
        return None  # 不符合任何条件的样本不处理

    return {
        'image': face_resize,
        'label': label,
        'bbox_offsets': (offset_x1, offset_y1, offset_x2, offset_y2),
        'landmark_offsets': (offset_px1, offset_py1, offset_px2, offset_py2, offset_px3, offset_py3, offset_px4, offset_py4, offset_px5, offset_py5)
    }
def process_annotation(face_size, anno_line, landmarks):
    """
    处理单行注释信息,生成正负样本。
    
    参数:
    anno_line (str): 一行注释信息,格式为 "image_filename x1 y1 w h"
    face_size (int): 生成的人脸图像尺寸
    landmarks (str): 关键点标注字符串
    
    返回:
    samples (list): 生成的样本列表
    """
    # 5个关键点
    _landmarks = landmarks.split()

    # 使用列表解析和解包一次性获取所有关键点的坐标
    landmarks = [float(x) for x in _landmarks[1:11]]

    # 解析注释行,获取图像文件名和人脸位置信息
    strs = parse_annotation_line(anno_line)
    image_filename = strs[0].strip()
    x1, y1, w, h = map(int, strs[1:])

    # 标签矫正
    x1, y1, x2, y2, w, h = adjust_bbox(x1, y1, w, h)
    boxes = [[x1, y1, x2, y2]]
    
    # 计算人脸中心点坐标
    cx = w / 2 + x1
    cy = h / 2 + y1

    # 最大边长
    max_side = max(w, h)
    
    # 打开图像文件
    image_filepath = os.path.join(IMG_PATH, image_filename)
    with Image.open(image_filepath) as img:
        # 解析出宽度和高度
        img_w, img_h = img.size
        # 生成候选的裁剪框
        samples = []
        for crop_box in generate_crop_boxes(cx, cy, max_side, img_w, img_h):
            # 处理每个候选裁剪框,生成正负样本
            sample = process_crop_box(img, face_size, max_side, crop_box, boxes, landmarks )
            if sample:
                samples.append(sample)
    
    return samples

def save_samples(samples, files, base_path, counters):
    """
    保存正负样本到文件中。
    
    参数:
    samples (list): 样本列表, 每个元素为一个字典, 包含 'image', 'label', 'bbox_offsets', 'landmark_offsets'
    files (dict): 包含正负样本输出文件的字典
    base_path (str): 输出文件的基础路径
    counters (dict): 样本计数器字典
    """
    for sample in samples:
        image = sample['image']
        label = sample['label']
        bbox_offsets = sample['bbox_offsets']
        landmark_offsets = sample['landmark_offsets']

        if label == 1:
            category = 'positive'
            counters['positive'] += 1
        elif label == 2:
            category = 'part'
            counters['part'] += 1
        else:
            category = 'negative'
            counters['negative'] += 1

        filename = f"{category}/{counters[category]}.jpg"
        image.save(os.path.join(base_path, filename))

        try:
            bbox_str = ' '.join(map(str, bbox_offsets))
            landmark_str = ' '.join(map(str, landmark_offsets))
            files[category].write(f"{filename} {label} {bbox_str} {landmark_str}\n")
        except IOError as e:
            print(f"Error writing to file: {e}")

def generate_samples(face_size, max_samples=-1):
    """
    生成指定大小的人脸样本,并保存到文件中。
    
    参数:
    face_size (int): 生成的人脸图像尺寸
    max_samples (int): 最大生成样本数量,设置为 -1 表示不限制
    """
    if not os.path.exists(DST_PATH):
        os.makedirs(DST_PATH)

    paths, base_path = create_directories(DST_PATH, face_size)
    # 新建标注文件
    files = open_label_files(base_path)

    # 样本计数
    counters = {'positive': 0, 'negative': 0, 'part': 0}

    # 读取标注信息
    with open(LANMARKS_PATH) as f:
        landmarks_list = f.readlines()
    with open(LABEL_PATH) as f:
        anno_list = f.readlines()

    for i, (anno_line, landmarks) in enumerate(zip(anno_list, landmarks_list)):
        print(f"positive:{counters['positive']}, \
                negative:{counters['negative']}, \
                part:{counters['part']}")
        
        # 跳过前两行
        if i < 2:
            continue
        
        # 如果处理了指定数量的样本,则退出循环
        if max_samples > 0 and i > max_samples:
            break
        
        # 处理单行标注信息,生成正负样本
        samples = process_annotation(
            face_size, anno_line, landmarks
        )
        
        # 保存正负样本到文件
        save_samples(
            samples,
            files, base_path, counters
        )

    for file in files.values():
        file.close()

def main():
    # 生成12×12的样本
    generate_samples(12, 1000)

    generate_samples(24, 1000)

    generate_samples(48, 1000)

if __name__ == "__main__":
    main()
