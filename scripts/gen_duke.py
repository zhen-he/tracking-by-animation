# resize images using gen_duke_processed.py
# get .pt training data using this file 


import os
import os.path as path
import glob
import argparse
import subprocess
from joblib import Parallel, delayed
import multiprocessing
import math
import numpy as np
import torch
import cv2 as cv
import sys
sys.path.append("modules")
import utils


parser = argparse.ArgumentParser()
parser.add_argument('--c', type=int, default=-1) # camera id
parser.add_argument('--v', type=int, default=0)
parser.add_argument('--metric', type=int, default=0)
arg = parser.parse_args()
assert arg.c == -1 or arg.c in range(1, 9), 'Invalid camera id.'

N = 1 if arg.metric == 1 else 24
T = 20
H = 108
W = 192
D = 3
train_ratio = 0 if arg.metric == 1 else 1
task = 'duke'

task_dir   = path.join('data', task)
image_dir  = path.join(task_dir, 'processed')
org_dir    = path.join(image_dir, 'input')
input_dir  = path.join(image_dir, 'input_roi')
bb_dir     = path.join(image_dir, 'bb')
bg_dir     = path.join(image_dir, 'bg')
camera_dir = '' if arg.c == -1 else 'camera'+str(arg.c)
metric_dir = 'metric' if arg.metric == 1 else ''
output_dir = path.join(task_dir, 'pt', camera_dir, metric_dir)
output_org_dir   = path.join(output_dir, 'org')
output_input_dir = path.join(output_dir, 'input')
output_bb_dir    = path.join(output_dir, 'bb')
output_bg_dir    = path.join(output_dir, 'bg')
if arg.v == 0:
    utils.rmdir(output_org_dir);   utils.mkdir(output_org_dir)
    utils.rmdir(output_input_dir); utils.mkdir(output_input_dir)
    utils.rmdir(output_bb_dir);    utils.mkdir(output_bb_dir)
    utils.rmdir(output_bg_dir);    utils.mkdir(output_bg_dir)


# Get train/test intervals
start_times = [5543, 3607, 27244, 31182, 1, 22402, 18968, 46766]
trainval = [49700, 227540]
test_easy = [263504, 356648]


# Get image_names and cam_ids
img_names_total = {'train': [], 'test': []}
cam_ids_total = {'train': [], 'test': []}
rg = range(1, 9) if arg.c == -1 else range(arg.c, arg.c+1)
for cam_id in rg:
    start_bias = start_times[cam_id-1] - 1
    train_start = trainval[0] - start_bias
    train_end = trainval[1] - start_bias
    test_start = test_easy[0] - start_bias
    test_end = test_easy[1] - start_bias
    input_img_dir = path.join(input_dir, 'camera'+str(cam_id))
    img_names = glob.glob(path.join(input_img_dir, '*.jpg')) # get filenames
    img_names = [path.basename(img_name) for img_name in img_names] # remove directory name
    if arg.metric == 0:
        img_names = list(filter(lambda f: train_start <= utils.get_num(f) <= train_end, img_names))
    else:
        img_names = list(filter(lambda f: test_start <= utils.get_num(f) <= test_end, img_names))
    img_names.sort(key=lambda f: utils.get_num(f)) # sort filename by number
    frame_num = len(img_names)
    print('frame number in camera' + str(cam_id) + ': ' + str(frame_num))
    train_frame_num = math.floor(frame_num * train_ratio)
    test_frame_num = math.floor(frame_num * (1 - train_ratio))
    img_names_total['train'] += img_names[0: train_frame_num]
    img_names_total['test']  += img_names[train_frame_num: train_frame_num+test_frame_num]
    cam_ids_total['train']   += [cam_id] * train_frame_num
    cam_ids_total['test']    += [cam_id] * test_frame_num
train_frame_num = len(img_names_total['train'])
test_frame_num = len(img_names_total['test'])
print('train frame number: ' + str(train_frame_num))
print('test frame number: ' + str(test_frame_num))
batch_nums = {
    'train': train_frame_num // (N * T),
    'test':  test_frame_num  // (N * T)
}
if arg.metric == 1:
    frame_map = [utils.get_num(img_name) for img_name in img_names_total['test']]
    utils.save_json(frame_map, path.join(output_dir, 'frame_map.json'))


core_num = 1 if arg.metric == 1 else multiprocessing.cpu_count()
print("Running with " + str(core_num) + " cores.")
def process_batch(split, ST, s, n):
    input_imgs, bg_imgs, org_imgs, bb_npys = [], [], [], []
    for t in range(0, T):
        i = n * ST + s * T + t
        img_name  = path.join('camera'+str(cam_ids_total[split][i]), img_names_total[split][i])

        input_img = cv.imread(path.join(input_dir, img_name))
        input_imgs.append(torch.from_numpy(input_img))

        bg_img = cv.imread(path.join(bg_dir, img_name))
        bg_imgs.append(torch.from_numpy(bg_img))

        org_img = cv.imread(path.join(org_dir, img_name))
        org_imgs.append(torch.from_numpy(org_img))

        if arg.metric == 0:
            bb_npy = np.load(path.join(bb_dir, path.splitext(img_name)[0]+'.npy'))
            bb_npys.append(torch.from_numpy(bb_npy))

    input_imgs = torch.stack(input_imgs, dim=0) # T * H * W * D
    bg_imgs    = torch.stack(bg_imgs, dim=0) # T * H * W * D
    org_imgs   = torch.stack(org_imgs, dim=0) # T * H * W * D

    if arg.metric == 0:
        bb_npys = torch.stack(bb_npys, dim=0) # T * O * 5
        return input_imgs, bg_imgs, org_imgs, bb_npys
    else:
        return input_imgs, bg_imgs, org_imgs


# Read image files and save them as torch tensors
with Parallel(n_jobs=core_num, backend="threading") as parallel:
    for split in ['train', 'test']:
        S = batch_nums[split]
        ST = S * T
        for s in range(0, S): # for each batch of sequences
            imgs_batch = parallel(delayed(process_batch)(split, ST, s, n) for n in range(0, N)) # N * 4 * T * H * W * D
            imgs_batch = list(zip(*imgs_batch)) # 4 * N * T * H * W * D

            input_batch_seq = torch.stack(imgs_batch[0], dim=0) # N * T * H * W * D
            bg_batch_seq    = torch.stack(imgs_batch[1], dim=0) # N * T * H * W * D
            org_batch_seq   = torch.stack(imgs_batch[2], dim=0) # N * T * H * W * D

            if arg.v == 1:
                for t in range(0, T):
                    utils.imshow(input_batch_seq[0, t], H*2, W*2, 'input')
                    utils.imshow(bg_batch_seq[0, t], H*2, W*2, 'bg', 1)
            else:
                filename = split + '_' + str(s) + '.pt'

                input_batch_seq = input_batch_seq.permute(0, 1, 4, 2, 3) # N * T * D * H * W
                torch.save(input_batch_seq, path.join(output_input_dir, filename))

                bg_batch_seq = bg_batch_seq.permute(0, 1, 4, 2, 3) # N * T * D * H * W
                torch.save(bg_batch_seq, path.join(output_bg_dir, filename))

                org_batch_seq = org_batch_seq.permute(0, 1, 4, 2, 3) # N * T * D * H * W
                torch.save(org_batch_seq, path.join(output_org_dir, filename))

                if arg.metric == 0:
                    bb_batch_seq = torch.stack(imgs_batch[3], dim=0) # N * T * O * 5
                    torch.save(bb_batch_seq, path.join(output_bb_dir, filename))

            print(split + ': ' + str(s+1) + ' / ' + str(S))


# save the data configuration
data_config = {
    'task': task,
    'train_batch_num': batch_nums['train'], 
    'test_batch_num': batch_nums['test'],
    'N': N,
    'T': T,
    'D': D,
    'H': H,
    'W': W
}
utils.save_json(data_config, path.join(output_dir, 'data_config.json'))