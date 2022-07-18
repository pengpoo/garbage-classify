# -*- coding: utf-8 -*-

import math
import numpy as np
from PIL import Image
from tensorflow.keras.utils import Sequence
from keras.applications.imagenet_utils import preprocess_input
from tools.cutout import Cutout
from keras.preprocessing.image import ImageDataGenerator

class GarbageClassifySequence(Sequence):
    """定义一个类，继承自 Sequence，用于对数据进行一些处理和
    从数据集中批量获取数据
    """
    def __init__(self, img_paths, labels, batch_size, img_size, is_train):
        assert len(img_paths) == len(labels)
        assert img_size[0] == img_size[1]

        self.x_y = np.hstack((np.array(img_paths).reshape(len(img_paths), 1), np.array(labels)))
        self.batch_size = batch_size
        self.img_size = img_size
        self.is_train = is_train
        # 训练集图片数据增强 Data Augmentation
        if self.is_train:
            train_datagen = ImageDataGenerator(
                rotation_range=30, # int 图片随机转动的角度
                width_shift_range=0.2, # float 宽度的某个比例 图片水平偏移的幅度
                height_shift_range=0.2, # float 高度的某个比例 图片竖直偏移的幅度
                shear_range=0.2, # float 剪切强度
                zoom_range=0.2, # float 或 list 随机缩放的幅度
                horizontal_flip=True, # 是否进行水平翻转
                vertical_flip=True, # 是否进行竖直翻转
                fill_mode='nearest' # 在变换时对超出边界的点的处理方法
            )
            self.train_datagen = train_datagen

    def __len__(self):
        return math.ceil(len(self.x_y) / self.batch_size)

    def preprocess_img(self, img_path):
        """图片预处理，(height, width) -> (224, 244)"""
        img = Image.open(img_path)
        img = img.resize((256, 256))
        img = img.convert('RGB') # PIL 读取的图片数据为 'RGBA'，多一个 transparency 通道
        img = np.array(img)
        img = img[16:16+224, 16:16+224]
        return img

    def cutout_img(self, img):
        cutout = Cutout(n_holes=1, length=40)
        img = cutout(img)
        return img

    def __getitem__(self, idx):
        # 获取batch中图片的路径
        batch_x = self.x_y[idx * self.batch_size:(idx + 1) * self.batch_size, 0]
        # 获取batch中图片的标签
        batch_y = self.x_y[idx * self.batch_size:(idx + 1) * self.batch_size, 1:]
        # 读取图片
        batch_x = np.array([self.preprocess_img(img_path=img_path) for img_path in batch_x])
        # 图片标签平滑归一化 图片共有40类
        batch_y = np.array(batch_y).astype(np.float32) * (1 - 0.005) + 0.005 / 40

        # 训练集图片数据增强
        if self.is_train:
            # 每一个batch中，随机选2/5的图片做cutout模拟遮挡处理，
            # 1/5的图片使用ImageDataGenerator做偏移、反转、剪切、转动等处理
            indexs = np.random.choice([0, 1, 2], batch_x.shape[0], replace=True, p=[0.4, 0.4, 0.2])
            mask_index = np.where(indexs==1)
            img_agu_index = np.where(indexs==2)

            if len(mask_index)>0:
                mask_batch_x = batch_x[mask_index]
                batch_x[mask_index] = np.array([self.cutout_img(img) for img in mask_batch_x])

            if len(img_agu_index)>0:
                img_agu_batch_x = batch_x[img_agu_index]
                img_agu_batch_y = batch_y[img_agu_index]

                train_data_generator = self.train_datagen.flow(img_agu_batch_x, img_agu_batch_y, batch_size=self.batch_size)
                (img_agu_batch_x, img_agu_batch_y) = train_data_generator.next()

                batch_x[img_agu_index] = img_agu_batch_x
                batch_y[img_agu_index] = img_agu_batch_y

        # 由于使用了迁移学习方法，选择了在 ImageNet 数据集上预训练的 ResNet50 模型
        # 按照 ImageNet 数据集图像处理方式处理图像
        batch_x = np.array([preprocess_input(img) for img in batch_x])

        return batch_x, batch_y

    def on_epoch_end(self):
        # 在每个 epoch 结束后将数据集顺序打乱
        np.random.shuffle(self.x_y)



