# -*- coding: utf-8 -*-

import re
import json
from tools.cust_utils import get_img_from_local_path, get_img_from_url
from train import add_custom_top_layers
import numpy as np


def load_model_weights(weights_dir):
    model = add_custom_top_layers()
    model.load_weights(weights_dir, by_name=True)
    return model


def predict(model, img_location):
    try:
        with open('./garbage_classify_v2/garbage_classify_rule.json', 'r', encoding='utf-8') as f:
            class_dict = json.load(f)
        if re.match(r'^https?:/{2}\w.+$', img_location):
            pred_imgs = get_img_from_url(img_location, 224)
            # print(type(pred_imgs)) -> list
            # print(np.array(pred_imgs).shape) -> (10, 224, 224, 3)
        else:
            pred_imgs = get_img_from_local_path(img_location, 224)

        if pred_imgs != 0:
            pred_num = 5  # 由于之前一张测试图片，会得到 10 个输入，pred_num 应小于等于 10
            predictions = np.array([0] * 40, dtype='float64')  # 或 predictions = [0]
            # 针对一张图片会做 pred_num 次预测
            for i in range(pred_num):
                pred_img = pred_imgs[i]
                pred_img = np.expand_dims(pred_img, axis=0)
                prediction = model.predict(pred_img)[0]
                predictions += prediction
            pred_label = np.argmax(predictions)
            pred_msg = class_dict[str(pred_label)]
            return pred_label, pred_msg
        else:
            print('ERROR READING IMAGE FILE')
            return False
    except Exception as e:
        print('running into error: ', e)


if __name__ == '__main__':
    model = load_model_weights('./output/best.h5')
    while True:
        try:
            img_location = input('请输入图片的绝对路径或 URL: ')
            print('图片地址为: ', img_location)
            pred_label, pred_msg = predict(model, img_location)
            print('==============+++==============')
            print('分类结果 calss: ', pred_label)
            print('该垃圾是：', pred_msg)
            print('==============+++==============')
        except Exception as e:
            print('running into error: ', e)

# 水瓶子 https://p6-tt.byteimg.com/origin/pgc-image/f4c2e53aab9645e78dda59e661af3bc6?from=pc
# 橘子皮 https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTAZm5MHQZKKVWlGKztyvCoGSbgpCLVR3Fbgw&usqp=CAU
# 橘子皮 https://img.ltn.com.tw/Upload/food/page/2019/02/01/190201-8604-1-xgoRZ.jpg
# 电池 https://www.nanfu.com/upload/at/image/20190904/1567563539516083eV2F.png
# 骨头 https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS0TdTXyaedhtipST4219TNwbuxnMAkrSiSBw&usqp=CAU
# 骨头 https://img.ljfl.wxwenku.com/static/cover_img_v2/1546.jpg
# https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQAfY5CO3GC7_Q-9cusHzoUTx9CxNl1kqMWPA&usqp=CAU
# 小黄鸭 https://img.alicdn.com/bao/uploaded/i1/1820777973/O1CN01kF9PNb28lgQtLrZpE_!!0-item_pic.jpg_300x300q90.jpg
# 药 https://img.alicdn.com/imgextra/i1/2200758668159/O1CN01CByoDW2A8s84oDCic_!!2200758668159.jpg
# 药 https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRYOZUk3Xe7D0ssSgwKUJzprZvsrsgDSKk2Fw&usqp=CAU