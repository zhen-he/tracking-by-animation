import os
import os.path as path
import glob
import argparse
import numpy as np
import torch
import sys
sys.path.append("modules")
import utils


parser = argparse.ArgumentParser()
parser.add_argument('--task', default='duke')
parser.add_argument('--subtask', default='')
parser.add_argument('--exp', default='tba')
parser.add_argument('--model', default='default')
arg = parser.parse_args()


conf_thresh = 0.5
duke_resize_scale = 0.1
duke_sample_step = 6
metric_dir = 'metric'
txt_name = 'duke.txt' if arg.task == 'duke' else 'metric.txt'

data_dir = path.join('data', arg.task, 'pt', arg.subtask, metric_dir)
result_metric_dir = path.join('result', arg.task, arg.subtask, arg.exp, arg.model, metric_dir)


pt_names = glob.glob(path.join(result_metric_dir, '*.pt')) # get filenames
pt_names = [path.basename(pt_name) for pt_name in pt_names] # remove directory name
pt_names.sort(key=lambda f: utils.get_num(f)) # sort filename by number
S = len(pt_names)

pt_tmp = torch.load(path.join(result_metric_dir, pt_names[0]))
N = pt_tmp.size(0)
T = pt_tmp.size(1)
O = pt_tmp.size(2)
assert pt_tmp.size(3) == 5, 'Wrong tensor size.'


# load results from all pt files
res = torch.Tensor(N, S, T, O, 5).zero_()
for s in range(0, S):
    pt = torch.load(path.join(result_metric_dir, pt_names[s])) # N * T * O * 5
    res[:,s].copy_(pt)
res = res.view(-1, O, 5) # NST * O * 5, F * O * 5
if arg.task == 'duke':
    cam = int(arg.subtask[6])
    res[:,:,0:4].div_(duke_resize_scale)
F = res.size(0)
print('length of pt files:', F)
# load frame mapping
if arg.task == 'duke':
    fmap = utils.load_json(path.join(data_dir, 'frame_map.json'))
    print('length of fmap:', len(fmap))


# convert and save results to txt file
confs = res[:,:,4] # F * O
confs = torch.cat((torch.Tensor(1, O).fill_(0), confs), dim=0) # (1 + F) * O
oids = torch.Tensor(O).fill_(0)
oid = 0 # object id, 0-based
with open(path.join(result_metric_dir, txt_name), "w") as file:
    for f in range(0, F):
        for o in range(0, O):
            conf_prev = confs[f,o]
            conf = confs[f+1,o]
            if conf >= conf_thresh:             # tracking
                if conf_prev < conf_thresh:     # new tracking
                    oids[o] = oid
                    oid += 1
                    if arg.task == 'duke':
                        file.write("%d,%d,%d,%.3f,%.3f,%.3f,%.3f\n" % (cam, oids[o], fmap[f], res[f,o,0], 
                                                                       res[f,o,1], res[f,o,2], res[f,o,3]))
                else:                           # maintain tracking
                    if arg.task == 'duke':      # interpolate sampled results
                        d = [(res[f,o,i] - res[f-1,o,i]) / duke_sample_step for i in range(0, 4)]
                        for i in range(duke_sample_step-1, -1, -1):
                            file.write("%d,%d,%d,%.3f,%.3f,%.3f,%.3f\n" % 
                                       (cam, oids[o], fmap[f]-i, res[f,o,0]-i*d[0], res[f,o,1]-i*d[1], 
                                        res[f,o,2]-i*d[2], res[f,o,3]-i*d[3]))
                if arg.task != 'duke':
                    file.write("%d,%d,%.3f,%.3f,%.3f,%.3f,-1,-1,-1,-1\n" %
                               (f+1, oids[o]+1, res[f,o,0], res[f,o,1], res[f,o,2], res[f,o,3]))
        print("Processing... %.1f%%" % ((f+1)/F*100))


# merge all txt files into one
if arg.task == 'duke' and arg.subtask == 'camera8':
    result_duke_metric_dir = path.join('result', arg.task, arg.exp, arg.model, metric_dir)
    utils.mkdir(result_duke_metric_dir)
    with open(path.join(result_duke_metric_dir, txt_name), "w") as fout:
        for i in range(1, 9):
            fin_name = path.join('result', arg.task, 'camera'+str(i), arg.exp, arg.model, metric_dir, txt_name)
            with open(fin_name) as fin:
                for line in fin:
                    fout.write(line)