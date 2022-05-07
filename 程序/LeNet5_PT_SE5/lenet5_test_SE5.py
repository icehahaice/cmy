# coding:utf-8
from __future__ import division
import argparse
import numpy as np
import cv2
import time
import sophon.sail as sail
from PIL import Image


def lenet5_preprocess(image):  # 预处理：归一化图片尺寸，并进行二值化处理
    '''resize and binary'''
    print("The size of input image", image.shape)
    image_GRAY = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(image_GRAY, (32, 32), interpolation=cv2.INTER_NEAREST)
    print("The size of resized_img", resized_img.shape)
    binary_img = cv2.adaptiveThreshold(resized_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 3, 10)
    x = binary_img / 255
    return x


def lenet5_postprocess(out):  # 后处理：利用模型推理结果，返回最大可能的数字
    value = list(out.values())
    value = np.squeeze(value)
    num = np.argmax(value)
    return num


def main():
    start_time = time.time()  # 开始时间
    # Note: pytorch read image use PIL when training
    img = Image.open(ARGS.input)
    img = np.array(img)
    # preprocess
    data = lenet5_preprocess(img)  # 调用预处理函数
    data = np.array([data])
    print("The shape of input of sail", data.shape)

    # sail core (inference)
    net = sail.Engine(ARGS.bmodel, ARGS.tpu_id, sail.IOMode.SYSIO)
    graph_name = net.get_graph_names()[0]  # get net name
    input_names = net.get_input_names(graph_name)  # get input names
    input_data = {input_names[0]: np.array([data], dtype=np.float32)}
    # 运行推理网络
    output = net.process(graph_name, input_data)
    # 调用后处理函数
    num_detect = lenet5_postprocess(output)
    end_time = time.time()  # 结束时间
    timer = end_time - start_time
    print("-" * 66)
    print("The number most likely: %d, time consuming: %.5f sec" % (num_detect, timer))
    print("-" * 66)


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description='for sail py test')
    PARSER.add_argument('--bmodel', default='./LeNet5_PT_bmodel/compilation.bmodel')
    PARSER.add_argument('--input', default='./5.jpg')
    PARSER.add_argument('--tpu_id', default=0, type=int, required=False)
    ARGS = PARSER.parse_args()
    main()
