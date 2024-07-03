import numpy as np
from tqdm import tqdm


def IOU(box, boxes, isMin=False):
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    # 两个左下角，取靠右上的
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])

    # 两个右下角，取靠左上的
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    # 求交集的边长，最短为0
    w = np.maximum(0, x2 - x1)
    h = np.maximum(0, y2 - y1)

    inter = w * h

    if isMin:
        return np.divide(inter, np.minimum(area, areas))
    else:
        return np.divide(inter, area + areas - inter)