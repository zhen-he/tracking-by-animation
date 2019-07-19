# generate bb_bg using bb_mask

import os
import os.path as path
import glob
import numpy as np
import argparse
import cv2 as cv
import sys
sys.path.append("modules")
import utils


# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--c', type=int, choices=range(1, 9)) # camera id
parser.add_argument('--v', type=int, default=0) # visualization
arg = parser.parse_args()
cam_id = arg.c


# specify directories
task_dir = 'data/duke'
camera_dir  = 'camera' + str(cam_id)
input_dir   = path.join(task_dir, 'frames', camera_dir)
gt_dir      = path.join(task_dir, 'ground_truth')
bb_mask_dir = path.join(gt_dir, 'bb_mask', camera_dir)
bb_bg_dir      = path.join(gt_dir, 'bb_bg', camera_dir)
if arg.v == 0:
    utils.rmdir(bb_bg_dir); utils.mkdir(bb_bg_dir)
step = 6
scl = 3
h, w = 108*scl, 192*scl # for visualization


# get filename
img_names = glob.glob(path.join(bb_mask_dir, '*.jpg')) # get filenames
img_names = [path.basename(img_name) for img_name in img_names] # remove directory name
img_names.sort(key=lambda f: utils.get_num(f)) # sort filename by number
frame_num = len(img_names)


# iteration
img_id_saved = -1
for i, img_name in enumerate(img_names, 1):

    # read input and bb_mask
    input_img = cv.imread(path.join(input_dir, img_name))
    bb_mask_img = cv.imread(path.join(bb_mask_dir, img_name), 0) # H * W
    bb_mask_img_binary = np.expand_dims((bb_mask_img > 127).astype(np.uint8), axis=2) # H * W * 1

    # synthesize bg
    img_id = int(path.splitext(img_name)[0])
    img_id_prev = img_id - step
    if img_id_prev == img_id_saved:
        bg_img = (1 - bb_mask_img_binary) * input_img + bb_mask_img_binary * bg_img_saved
    else:
        input_img_prev = cv.imread(path.join(input_dir, str(img_id_prev)+'.jpg'))
        bg_img = (1 - bb_mask_img_binary) * input_img + bb_mask_img_binary * input_img_prev

    # visualize
    if arg.v == 1:
        utils.imshow(input_img, h, w, 'input_img')
        utils.imshow(bb_mask_img, h, w, 'bb_mask_img')
        utils.imshow(np.absolute(input_img.astype(np.int16)-bg_img.astype(np.int16)).astype(np.uint8), h, w, 'diff_img')
        utils.imshow(bg_img, h, w, 'bg_img', 1)
    else:
        cv.imwrite(path.join(bb_bg_dir, img_name), bg_img)

    # save
    img_id_saved = img_id
    bg_img_saved = bg_img
    print(img_name, i, frame_num)
