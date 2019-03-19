# use ROIs, resize images, get input, bb, and bg

import os
import os.path as path
import glob
import argparse
import math
import numpy as np
import torch
import cv2 as cv
import sys
sys.path.append("modules")
import utils


parser = argparse.ArgumentParser()
parser.add_argument('--c', type=int, choices=range(1, 9)) # camera id
parser.add_argument('--v', type=int, default=0)
arg = parser.parse_args()
cam_id = arg.c
#
h = 108
w = 192
scl = 2 # for visualization
morph_kernel_size = 3


# Specify directories
task_dir    = 'data/duke'
camera_dir  = 'camera' + str(cam_id)
input_dir   = path.join(task_dir, 'frames', camera_dir)
fg_mask_dir = path.join(task_dir, 'imbs', 'fg_mask', camera_dir)
imbs_bg_dir = path.join(task_dir, 'imbs', 'bg', camera_dir)
gt_dir      = path.join(task_dir, 'ground_truth')
bb_dir      = path.join(gt_dir, 'bb', camera_dir)
bb_bg_dir   = path.join(gt_dir, 'bb_bg', camera_dir)
roi_dir     = path.join(task_dir, 'calibration')
#
output_dir           = path.join(task_dir, 'processed')
output_input_dir     = path.join(output_dir, 'input', camera_dir)
output_input_roi_dir = path.join(output_dir, 'input_roi', camera_dir)
output_bb_dir        = path.join(output_dir, 'bb', camera_dir)
output_bg_dir        = path.join(output_dir, 'bg', camera_dir)
if arg.v == 0:
    utils.rmdir(output_input_dir);     utils.mkdir(output_input_dir)
    utils.rmdir(output_input_roi_dir); utils.mkdir(output_input_roi_dir)
    utils.rmdir(output_bb_dir);        utils.mkdir(output_bb_dir)
    utils.rmdir(output_bg_dir);        utils.mkdir(output_bg_dir)


# Get train/test intervals
start_times = [5543, 3607, 27244, 31182, 1, 22402, 18968, 46766]
trainval = [49700, 227540]
test_easy = [263504, 356648]
start_bias = start_times[cam_id-1] - 1
train_start = trainval[0] - start_bias
train_end = trainval[1] - start_bias
test_start = test_easy[0] - start_bias
test_end = test_easy[1] - start_bias


# Morphological closing operation for fg mask post-processing
morph_kernel = cv.getStructuringElement(cv.MORPH_RECT, (morph_kernel_size, morph_kernel_size))
pad_width = morph_kernel_size // 2 + 1
def morph_close(img):
    img_pad = cv.copyMakeBorder(img, pad_width, pad_width, pad_width, pad_width, cv.BORDER_CONSTANT, 0)
    img_pad_out = cv.morphologyEx(img_pad, cv.MORPH_CLOSE, morph_kernel)
    img_out = img_pad_out[pad_width:pad_width+h, pad_width:pad_width+w]
    # _, img_out = cv.threshold(img_pad_out[pad_width:pad_width+h, pad_width:pad_width+w], 127, 255, cv.THRESH_BINARY)
    return img_out


# get filename
img_names = glob.glob(path.join(fg_mask_dir, '*.jpg')) # get filenames
img_names = [path.basename(img_name) for img_name in img_names] # remove directory name
img_names.sort(key=lambda f: utils.get_num(f)) # sort filename by number
frame_num = len(img_names)


# Initialization
roi_img = cv.imread(path.join(roi_dir, 'roi'+str(cam_id)+'_modified.jpg'), 0) # H * W
roi_img = (roi_img > 127).astype('uint8')
roi_img_r = roi_img.reshape((roi_img.shape[0], roi_img.shape[1], 1)) # H * W * 1
for img_name in img_names:
    img_id = int(path.splitext(img_name)[0])
    if train_start <= img_id <= train_end:
        bb_filename = path.join(bb_dir, str(img_id)+'.npy')
        if path.isfile(bb_filename):
            bb = np.load(bb_filename)
            zeros = bb.copy()
            zeros.fill(0)
            break


# Iteration
for i, img_name in enumerate(img_names, 1):
    img_id = int(path.splitext(img_name)[0])
    if train_start <= img_id <= train_end or test_start <= img_id <= test_end:
        # read input images
        input_img     = cv.imread(path.join(input_dir, img_name)) # H * W * D
        input_roi_img = input_img * roi_img_r # H * W * D
        input_img     = cv.resize(input_img, (w, h)) # h * w * D
        input_roi_img = cv.resize(input_roi_img, (w, h)) # h * w * D
        
        # for training
        if train_start <= img_id <= train_end:
            bb_filename = path.join(bb_dir, str(img_id)+'.npy')
            if path.isfile(bb_filename):
                bb = np.load(bb_filename)
                bb_bg_img = cv.imread(path.join(bb_bg_dir, img_name)) * roi_img_r
                bg_img = cv.resize(bb_bg_img, (w, h))
            else:
                bb = zeros
                bg_img = input_roi_img

        # for test
        elif test_start <= img_id <= test_end:
            imgs_bg_img = cv.imread(path.join(imbs_bg_dir, img_name)) # H * W * D
            imgs_bg_img = cv.resize(imgs_bg_img, (w, h)) # h * w * D
            fg_mask_img = cv.imread(path.join(fg_mask_dir, img_name), 0) * roi_img # H * W
            fg_mask_img = morph_close(cv.resize(fg_mask_img, (w, h))) # h * w
            fg_mask_img_binary = np.expand_dims((fg_mask_img > 127).astype(np.uint8), axis=2) # h * w * 1
            bg_img = (1 - fg_mask_img_binary) * input_img + fg_mask_img_binary * imgs_bg_img

        # visualize or save
        if arg.v == 1:
            utils.imshow(input_img, h*scl, w*scl, 'input_img')
            utils.imshow(input_roi_img, h*scl, w*scl, 'input_roi_img')
            utils.imshow(bg_img, h*scl, w*scl, 'bg_img')
            utils.imshow(np.absolute(input_roi_img.astype('int16')-bg_img.astype('int16')).astype('uint8'), 
                         h*scl, w*scl, 'diff_img', 1)
        else:
            cv.imwrite(path.join(output_input_dir, img_name), input_img)
            cv.imwrite(path.join(output_input_roi_dir, img_name), input_roi_img)
            cv.imwrite(path.join(output_bg_dir, img_name), bg_img)
            if train_start <= img_id <= train_end:
                np.save(path.join(output_bb_dir, str(img_id)+'.npy'), bb)

    print(img_name, i, frame_num)