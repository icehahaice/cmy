import torch
import cv2
import numpy as np
import time


def read_data(im_path):
    img = cv2.imread(im_path, 0)
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_NEAREST)
    binary_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 3, 10)
    x = binary_img / 255
    x = np.expand_dims(x, axis=0)
    x = np.expand_dims(x, axis=0)
    x = x.astype(np.float32)
    return x


def main(img):
    loaded = torch.jit.load('./model/lenet.zip')

    input_tensor = torch.from_numpy(img)
    ret = loaded(input_tensor)
    ret = ret.detach().numpy()
    index = ret.argmax()
    print("-" * 66)
    print("The number most likely: %d, score: %.3f" % (index, ret[0][index]))
    print("-" * 66)


if __name__ == '__main__':
    start = time.time()
    img = read_data("./data/6n.jpg")
    main(img)
    end = time.time()
    print("运行时间:%.2f秒" % (end - start))
