# -*- coding: utf-8 -*-

import numpy as np


class Cutout(object):
    """随机选择一个固定大小的正方形区域，
    用 0 填充。模拟遮挡，提高泛化能力
    Args:
        n_holes (int): 需要 cutout 的正方形区域个数
        length (int): cutout 正方形的边长（像素点）
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.shape[0]
        w = img.shape[1]
        c = img.shape[2]

        mask = np.ones((h, w, c), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y-self.length//2, 0, h)
            y2 = np.clip(y+self.length//2, 0, h)
            x1 = np.clip(x-self.length//2, 0, w)
            x2 = np.clip(x+self.length//2, 0, w)
            mask[y1:y2, x1:x2] = 0

        img = img * mask

        return img