import os
import os.path as path
import shutil
import torch
import cv2 as cv
import json
import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt


def imshow(img, height=None, width=None, name='img', delay=1):
    # img: H * W * D
    if torch.is_tensor(img):
        img = img.cpu().numpy()
    h = img.shape[0] if height == None else height
    w = img.shape[1] if width == None else width
    cv.namedWindow(name, cv.WINDOW_NORMAL)
    cv.resizeWindow(name, w, h)
    cv.imshow(name, img)
    cv.waitKey(delay)


def imwrite(img, name='img'):
    # img: H * W * 
    img = (img * 255).byte().cpu().numpy()
    cv.imwrite(name + '.jpg', img)


def imresize(img, height, width):
    # img: H * W * D
    is_torch = False
    if torch.is_tensor(img):
        img = img.cpu().numpy()
        is_torch = True
    img_resized = cv.resize(img, (width, height))
    if is_torch:
        img_resized = torch.from_numpy(img_resized)
    return img_resized


def heatmap(img, cmap='hot'):
    cm = plt.get_cmap(cmap)
    cimg = cm(img.cpu().numpy())
    cimg = torch.from_numpy(cimg[:, :, :3])
    cimg = torch.index_select(cimg, 2, torch.LongTensor([2, 1, 0])) # convert to BGR for opencv
    return cimg


def save_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)


def load_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data


def mkdir(dir):
    if not path.exists(dir):
        os.makedirs(dir)


def rmdir(dir):
    if path.exists(dir):
        print('Directory ' + dir + ' is removed.')
        shutil.rmtree(dir)


def get_num(s):
    # return int(path.splitext(s)[0])
    return int(''.join(filter(str.isdigit, s)))


def getGaussianKernel(ksize, sigma):
    k = np.zeros((ksize, ksize))
    k[ksize//2, ksize//2] = 1
    k = ndimage.filters.gaussian_filter(k, sigma, mode='constant')
    k = k / k.sum()
    k = torch.from_numpy(k).view(1, 1, ksize, ksize).float()
    return k