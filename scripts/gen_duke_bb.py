# generate bb and bb_mask

import scipy.io as sio
import os
import os.path as path
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


# specify directories and parameters
task_dir = 'data/duke'
camera_dir  = 'camera' + str(cam_id)
fg_mask_dir = path.join(task_dir, 'imbs', 'fg_mask', camera_dir)
gt_dir      = path.join(task_dir, 'ground_truth')
bb_dir      = path.join(gt_dir, 'bb', camera_dir)
bb_mask_dir = path.join(gt_dir, 'bb_mask', camera_dir)
if arg.v == 0:
    utils.rmdir(bb_dir);      utils.mkdir(bb_dir)
    utils.rmdir(bb_mask_dir); utils.mkdir(bb_mask_dir)
#
H, W = 108, 192
scl = 1080/H
zeta_s = 0.55
zeta_r = [1.25, 0.3]
h = 23
w = 9
O = 12 # maximum object number
step = 6
scl_v = 3
hv, wv = H*scl_v, W*scl_v # for visualization


# get ground truth data
gt_total = sio.loadmat(path.join(gt_dir, 'trainval.mat'))
gt_total = list(gt_total['trainData']) # N * 11
#      0,  1,     2,    3,   4,     5,      6,      7,      8,     9,     10,    11,    12,      13,      14
# camera, ID, frame, left, top, width, height, worldX, worldY, feetX, feetyY, scale, ratio, trans_x, trans_y
gt = list(filter(lambda f: f[0] == cam_id, gt_total))
gt.sort(key=lambda f: f[2]) # sort by frames


# normalize bb values to [-1, 1]
gt = np.asarray(gt)
left   = gt[:, 3:4] / scl
top    = gt[:, 4:5] / scl
width  = gt[:, 5:6] / scl
height = gt[:, 6:7] / scl
center_x = left + (width-1)/2
center_y = top + (height-1)/2
trans_x = -(center_x-1)/(W-1) * 2 + 1  # [-1, 1]
trans_y = -(center_y-1)/(H-1) * 2 + 1  # [-1, 1]
scale = (np.sqrt(height*width/h/w) - 1) / zeta_s
ratio = (height/width/h*w - zeta_r[0]) / zeta_r[1]
gt = list(np.concatenate((gt, scale, ratio, trans_x, trans_y), axis=1))


# function to get process bbs
def process_bb(frame, tids, bbs_normed, bbs_normed_all, bbs_draw):
    # classify tids
    set_tid      = set(bbs_normed_all.keys())
    set_tid_prev = set(bbs_normed.keys())
    tid_inter  = list(set_tid & set_tid_prev) # intersected tids
    tid_new    = list(set_tid - set_tid_prev) # new tids
    tid_disapp = list(set_tid_prev - set_tid) # disappeared tids

    # update existing bbs
    for tid in tid_inter:   
        bbs_normed[tid] = bbs_normed_all[tid]

    # add new bbs
    for tid in tid_new:
        if len(tids) < O:      
            oids_free = list(set(range(1, O+1)) - set(tids.keys()))
            tids[oids_free[0]] = tid
            bbs_normed[tid] = bbs_normed_all[tid]

    # delete disappeared bbs
    tids = {k: v for k, v in tids.items() if v not in tid_disapp}
    bbs_normed = {k: v for k, v in bbs_normed.items() if k not in tid_disapp}
    
    # get a bb_normed matrix, where the 1st column denotes the bb exsitence
    bb_mat = np.zeros((O, 5), 'float')
    for oid in range(1, O+1):
        if oid in tids.keys():  # exist
            tid = tids[oid]
            bb_mat[oid-1, 0] = 1  
            bb_mat[oid-1, 1:5] = bbs_normed[tid]
        else:                   # not exist
            bb_mat[oid-1] = 0  
    
    # draw bb masks
    bb_mask = np.zeros((1080, 1920), 'uint8')
    for bb_draw in bbs_draw:
        cv.fillConvexPoly(bb_mask, bb_draw, 255)

    # visualize & save
    if arg.v == 1:
        fg_mask = cv.imread(path.join(fg_mask_dir, str(frame)+'.jpg'), 0)
        fg_mask_vis = fg_mask//2 + bb_mask//2
        print(bb_mat)
        utils.imshow(bb_mask, hv, wv, 'bb_mask')
        utils.imshow(fg_mask_vis, hv, wv, 'fg_mask_vis', 1)
    else:
        np.save(path.join(bb_dir, str(frame)), bb_mat)
        cv.imwrite(path.join(bb_mask_dir, str(frame)+'.jpg'), bb_mask)

    return tids, bbs_normed


# iteration
tids = {} # {oid: tid}
bbs_normed = {}  # {tid: bb_normed}
frame_prev = -1
first_frame = True
for i, v in enumerate(gt, 1):
    frame = int(v[2])
    if (frame - 1) % step == 0:
        # get normalized bbs
        tid = int(v[1])
        bb_normed = v[11:15]

        # get bbs for drawing masks
        bb_l, bb_t, bb_w, bb_h = v[3:7]
        bb_r = bb_l + bb_w - 1
        bb_b = bb_t + bb_h - 1
        bb_draw = np.array([[bb_l, bb_t], [bb_r, bb_t], [bb_r, bb_b], [bb_l, bb_b]], 'int32')

        # process bbs
        if frame != frame_prev:  # a new frame
            if not first_frame:
                tids, bbs_normed = process_bb(frame_prev, tids, bbs_normed, bbs_normed_all, bbs_draw)
            # save & initialize
            bbs_normed_all = {tid: bb_normed}
            bbs_draw = [bb_draw]
            frame_prev = frame
            first_frame = False
        else:                    # the same frame
            bbs_normed_all[tid] = bb_normed
            bbs_draw += [bb_draw]
    print(i, len(gt))
# for the last frame
tids, bbs_normed = process_bb(frame_prev, tids, bbs_normed, bbs_normed_all, bbs_draw)
