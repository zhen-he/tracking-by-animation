import os
import os.path as path
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append("modules")
import utils


# args
parser = argparse.ArgumentParser()
parser.add_argument('--task', default='duke', choices=['mnist', 'sprite', 'duke'])
parser.add_argument('--subtask', default='')
parser.add_argument('--exps', nargs='+', default=['tba'])
parser.add_argument('--model', default='default')
parser.add_argument('--train', type=int, default=1)
parser.add_argument('--smooth_size', type=int, default=1)
parser.add_argument('--step', type=int, default=1) # iteration sample step
parser.add_argument('--xepoch', type=int, default=0) # show epochs instead of iterations in x-axis
o = parser.parse_args()


# global variables
colors = sns.color_palette("Set2", 10)
# lines = {
#     'air':        ['AIR',       '*', colors[0]],
#     'tba_no_occ': ['TBA-noOcc', 's', colors[1]],
#     'tba_no_mem': ['TBA-noMem', '^', colors[2]],
#     'tba_no_att': ['TBA-noAtt', 'v', colors[3]],
#     'tba_no_rep': ['TBA-noRep', 'o', colors[4]],
#     'tba':        ['TBA',       '<', colors[5]],
#     'xx1':        ['xx1',       '>', colors[6]],
#     'xx2':        ['xx2',       'v', colors[7]],
#     'xx3':        ['xx3',       '.', colors[8]],
#     'xx4':        ['xx4',       '+', colors[9]]
# }
lines = {
    'tba':         ['TBA',        '',  colors[6]],
    'tbac':        ['TBAc',       '.', colors[5]],
    'tbac_no_occ': ['TBAc-noOcc', 'v', colors[2]],
    'tbac_no_att': ['TBAc-noAtt', 'o', colors[1]],
    'tbac_no_mem': ['TBAc-noMem', '^', colors[3]],
    'tbac_no_rep': ['TBAc-noRep', 's', colors[4]],
    'xx1':         ['xx1',        '',  colors[7]],
    'xx2':         ['xx2',        '',  colors[8]],
}
plt.rc('lines', linewidth=3, markersize=10)
plt.rc('axes', labelsize=12)
plt.rc('xtick', labelsize=10.5)
plt.rc('ytick', labelsize=10.5)
plt.rc('legend', fontsize=12)

split = 'train' if o.train == 1 else 'val'
result_dir = path.join('result', o.task, o.subtask)
smooth_pad = o.smooth_size // 2


def main():
    results, train_batch_nums = load_multi_result()
    num = min([len(v[0]) for v in results.values()])

    plt.figure(1)
    handles = []
    for exp, result in results.items():
        x = [result[0][p] for p in range(0, num, o.step)]
        if o.xepoch == 1:
            x = [v/train_batch_nums[exp] for v in x]
        y = result[1][0:num]
        if o.smooth_size > 1:
            y = [y[0]]*smooth_pad + y + [y[-1]]*smooth_pad
            y = [sum(y[i-smooth_pad:i+smooth_pad+1])/(smooth_pad*2+1) for i in range(smooth_pad, smooth_pad+num)]
        y = [y[p] for p in range(0, num, o.step)]
        h, = plt.plot(x, y, label=lines[exp][0], marker=lines[exp][1], color=lines[exp][2])
        handles = [h] + handles
    plt.legend(handles = handles)
    xlabel = 'Iteration' if o.xepoch == 0 else 'Epoch'
    plt.xlabel(xlabel)
    plt.ylabel('Validation Loss')
    plt.show()


def load_single_result(exp):
    result = utils.load_json(path.join(result_dir, exp, o.model, 'sp_latest.json'))
    loss_pairs = result['benchmark'][split + '_loss']
    train_batch_num = result['o']['train_batch_num']
    return list(map(list, zip(*loss_pairs))), train_batch_num


def load_multi_result():
    results, train_batch_nums = {}, {}
    for exp in o.exps:
        results[exp], train_batch_nums[exp] = load_single_result(exp)
    return results, train_batch_nums


main()