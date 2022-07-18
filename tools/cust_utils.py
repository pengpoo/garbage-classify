# -*- coding: utf-8 -*-


import os
import codecs
import random
from PIL import Image
from keras.applications.imagenet_utils import preprocess_input
from glob import glob
import requests
from io import BytesIO
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from tools.garbage_sequence import GarbageClassifySequence
import numpy as np


def data_flow(data_dir, batch_size, num_classes, input_size):
    """
    参数：
    data_dir(str): 图片文件加路径
    batch_size(int): 批量大小
    num_classes(int): 分类任务类别个数
    input_size(int): 输入图片大小
    return: 训练集和验证集的 DataLoader 实例
    """
    label_files = glob(os.path.join(data_dir, '*.txt'))
    random.shuffle(label_files)
    img_paths = []
    labels = []
    for idx, file_path in enumerate(label_files):
        with codecs.open(file_path, 'r', 'utf-8') as f:
            line = f.readline()
        line_split = line.strip().split(', ')
        if len(line_split) != 2:
            print(f"{os.path.basename(file_path)} seems like a bad label file")
            continue
        img_file_name = line_split[0]
        label = int(line_split[1])
        img_paths.append(os.path.join(data_dir, img_file_name))
        labels.append(label)

    labels = np_utils.to_categorical(labels, num_classes=num_classes)
    train_img_paths, validation_img_paths,\
    train_labels, validation_labels = train_test_split(img_paths,
                                                       labels,
                                                       test_size=0.1,
                                                       random_state=233)
    print(f"total samples: {len(img_paths)}, training set samples: {len(train_img_paths)}, "
          f"validation set samples: {len(validation_img_paths)}")
    # 生成训练集验证集 DataLoader 实例
    train_sequence = GarbageClassifySequence(img_paths=train_img_paths,
                                             labels=train_labels,
                                             batch_size=batch_size,
                                             img_size=[input_size, input_size],
                                             is_train=True)
    validation_sequence = GarbageClassifySequence(img_paths=validation_img_paths,
                                                  labels=validation_labels,
                                                  batch_size=batch_size,
                                                  img_size=[input_size, input_size],
                                                  is_train=False)
    return train_sequence, validation_sequence


def get_img_from_local_path(img_path, img_size=224):
    """从本地路径中读取处理图片
    参数：
    img_path(str): 图片的本地路径
    img_size(int): 使用在 ImageNet 上预训练的 ResNet50 网络，输入图片的高度和宽度应为 224
    先将图片高度和宽度调整为(256, 256)
    再将图片边缘的一些像素点丢弃，得到高度和宽度为 224 的图片输入
    随机进行了 10 次裁剪，对于一张图片得到 10 个裁剪后的结果
    """
    try:
        img = Image.open(img_path)
        img = img.resize((256, 256))
        img = img.convert('RGB')
        img = np.array(img)  # (height, width, channel) -> (256, 256, 3)
        imgs = []
        for _ in range(10):
            i = random.randint(0, 32)
            j = random.randint(0, 32)
            img_cc = img[i:i + img_size, j:j + img_size]
            img_cc = preprocess_input(img_cc)
            imgs.append(img_cc)
        return imgs
    except Exception as e:
        print('running into error: ', e)
        return 0


def get_img_from_url(img_url, img_size=224):
    """从 URL 中读取处理图片
    参数：
    img_url(str): 图片的 URL
    img_size(int): 使用在 ImageNet 上预训练的 ResNet50 网络，输入图片的高度和宽度应为 224
    """
    try:
        response = requests.get(img_url)
        img = Image.open(BytesIO(response.content))
        img = img.resize((256, 256))
        img = img.convert('RGB')
        img = np.array(img)
        imgs = []
        for _ in range(10):
            i = random.randint(0, 32)
            j = random.randint(0, 32)
            img_cc = img[i:i+img_size, j:j+img_size]
            img_cc = preprocess_input(img_cc)
            imgs.append(img_cc)
        return imgs
    except Exception as e:
        print('running into error: ', e)
        return 0


def load_test_data(FLAGS):
    """加载测试集数据
    """
    pass
