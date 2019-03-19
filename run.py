import time
import os
import os.path as path
import argparse
import numpy as np
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import cv2 as cv
import modules.utils as utils
from modules.net import Net


# Parse arguments
parser = argparse.ArgumentParser()
# Task
parser.add_argument('--task', default='duke', choices=['mnist', 'sprite', 'duke'],
                    help="Choose a task from mnist/sprite/duke")
parser.add_argument('--subtask', default='',
                    help="Choose a subtask for duke, from camera1 to camera8 ('' for all cameras)")
parser.add_argument('--exp', default='tba',
                    help="Choose an experiment from tba/tbac/tbac_no_occ/tbac_no_att/tbac_no_mem/tbac_no_rep")
parser.add_argument('--model', default='default',
                    help="Choose a model with different hyper-parameters (specified in 'modules/model_config.json')") 
parser.add_argument('--train', type=int, default=1, choices=[0, 1],
                    help="Choose to train (1) or test (0) the model")
parser.add_argument('--metric', type=int, default=0, choices=[0, 1],
                    help="Choose whether to gnerate results to measure the tracking performance")
parser.add_argument('--init_model', default='',
                    help="Recover training from a checkpoint, e.g., 'sp_latest.pt', 'sp_3000.pt'")
# Model settings
parser.add_argument('--r', type=int, default=1, choices=[0, 1],
                    help="Choose whether to remember the recurrent state from the previous sequence")
# Training
parser.add_argument('--epoch_num', type=int, default=500,
                    help="The number of training epoches")
parser.add_argument('--reset_interval', type=float, default=0.01,
                    help="Set how to reset the recurrent state, \
                    (-inf, 0): do not reset, [0, 1): the probability to reset, [1, inf): time steps to reset")
parser.add_argument('--print_interval', type=int, default=1,
                    help="Iterations to print training messages") 
parser.add_argument('--train_log_interval', type=int, default=100,
                    help="Iterations to log training messages") 
parser.add_argument('--save_interval', type=int, default=100,
                    help="Iterations to save checkpoints (will be overwitten)") 
parser.add_argument('--validate_interval', type=int, default=1000,
                    help="Iterations to validate model and save checkpoints") 
# Optimization
parser.add_argument('--lr', type=float, default=5e-4,
                    help="Learning rate")
parser.add_argument('--lr_decay_factor', type=float, default=1,
                    help="Learning rate decay factor")
parser.add_argument('--grad_clip', type=float, default=5,
                    help='Gradient clipping value')
# Others
parser.add_argument('--s', type=int, default=0, choices=[0, 1],
                    help='Benchmark the speed')
parser.add_argument('--v', type=int, default=0, choices=[0, 1, 2],
                    help="0: no visualization, 1: show on screen, 2: save to disk")
o = parser.parse_args()


metric_dir = 'metric' if o.metric == 1 else ''
data_dir = path.join('data', o.task, 'pt', o.subtask, metric_dir)
result_dir = path.join('result', o.task, o.exp, o.model)
utils.mkdir(result_dir)
result_file_header = path.join(result_dir, 'sp_')
if o.metric == 1:
    o.train = 0
    if o.v == 0:
        o.result_metric_dir = path.join('result', o.task, o.subtask, o.exp, o.model, metric_dir);
        utils.rmdir(o.result_metric_dir); utils.mkdir(o.result_metric_dir)
    elif o.v == 2:
        o.pic_dir = path.join('pic', o.task, o.subtask, o.exp, o.model, metric_dir)
        utils.rmdir(o.pic_dir); utils.mkdir(o.pic_dir)


# Initialize configuration variables
# Load data configuration
data_config = utils.load_json(path.join(data_dir, 'data_config.json'))
for k, v in data_config.items():
    vars(o)[k] = v
print("Task: " + o.task)
# Load experiment configuration
o.exp_config = utils.load_json('modules/exp_config.json')[o.exp]
print("Experiment: " + o.exp)
print("Configuration: ", o.exp_config)
# Load model configuration
model_config = utils.load_json('modules/model_config.json')[o.task]["default"]
if o.model != 'default':
    model_config.update(utils.load_json('modules/model_config.json')[o.task][o.model])
for k, v in model_config.items():
    vars(o)[k] = v
print("Model: " + o.model)
# Set other hyper-parameters
if o.task in ['mnist', 'sprite']:
    o.bg = 0
elif o.task in ['duke']:
    o.bg = 1
# C in 3D shape
o.dim_C3_1 = o.cnn['out_sizes'][-1][0]
o.dim_C3_2 = o.cnn['out_sizes'][-1][1]
o.dim_C3_3 = o.cnn['conv_features'][-1]
o.dim_h_o = o.dim_C3_3 * 4
# C in 2D shape
if "no_att" in o.exp_config:
    o.dim_C2_1 = 1
    o.dim_C2_2 = o.dim_C3_1 * o.dim_C3_2 * o.dim_C3_3
else:
    o.dim_C2_1 = o.dim_C3_1 * o.dim_C3_2
    o.dim_C2_2 = o.dim_C3_3
if "no_occ" in o.exp_config:
    o.dim_y_l = 1
o.dim_y_e = 1
o.dim_y_p = 4
o.dim_Y_s = 1 * o.h * o.w
o.dim_Y_a = o.D * o.h * o.w
# Set GPU number
o.G = 1 if o.metric == 1 else torch.cuda.device_count() # get GPU number
assert o.N % o.G == 0, 'Please ensure the mini-batch size can be divided by the GPU number'
o.n = round(o.N / o.G)
print("Total batch size: %d, GPU number: %d, GPU batch size: %d" % (o.N, o.G, o.n))


# Initialize the model and benchmark
net = Net(o).cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=o.lr, betas=(0.9, 0.99), weight_decay=1e-6)
benchmark = {'train_loss': [], 'val_loss': [], 'i_start': 0}
if o.init_model == '':
    pass
    # f_bias = 1
    # net.tracker_array.ntm.ntm_cell.rnn_cell.bias_ih.data[o.dim_h_o: o.dim_h_o * 2].fill_(f_bias/2)
    # net.tracker_array.ntm.ntm_cell.rnn_cell.bias_hh.data[o.dim_h_o: o.dim_h_o * 2].fill_(f_bias/2)
else:
    f = path.join(result_dir, o.init_model)
    savepoint = torch.load(f)
    net.load_state_dict(savepoint['net_states'])
    # if 'optim_states' in savepoint.keys():
    #     optimizer.load_state_dict(savepoint['optim_states'])
    benchmark = savepoint['benchmark']
    print('Model is initialized from ' + f)
param_num = sum([param.data.numel() for param in net.parameters()])
print('Parameter number: %.3f M' % (param_num / 1024 / 1024))

# o.test_batch_num = o.train_batch_num
# Data loader
def load_data(batch_id, split):
    volatile = False if split == 'train' else True
    # split = 'train'
    filename = split + '_' + str(batch_id) + '.pt'
    kwargs = {}
    X_seq = torch.load(path.join(data_dir, 'input', filename))
    X_seq = Variable(X_seq.float().cuda().div_(255), volatile=volatile)
    if o.bg == 1:
        X_bg_seq = torch.load(path.join(data_dir, 'bg', filename))
        kwargs['X_bg_seq'] = Variable(X_bg_seq.float().cuda().div_(255), volatile=volatile)
        if o.metric == 1:
            X_org_seq = torch.load(path.join(data_dir, 'org', filename))
            kwargs['X_org_seq'] = Variable(X_org_seq.float().cuda().div_(255), volatile=volatile)
    return X_seq, kwargs


# Forward function
def forward(X_seq, **kwargs):
    start_time = time.time()
    loss = net(X_seq, **kwargs)
    elapsed_time = time.time() - start_time
    return loss, elapsed_time


# Backward function
def backward(loss):
    start_time = time.time()
    optimizer.zero_grad()
    loss.backward()
    if o.grad_clip > 0:
        # params = list(filter(lambda p: p.grad is not None, net.parameters()))
        # for param in params:
        #     param.grad.data.clamp_(-o.grad_clip, o.grad_clip)
        nn.utils.clip_grad_norm(net.parameters(), o.grad_clip)
    optimizer.step()
    elapsed_time = time.time() - start_time
    return elapsed_time


# The forward and backward passes for an iteration
def run_batch(batch_id, split):
    X_seq, kwargs = load_data(batch_id, split)
    loss, forward_time = forward(X_seq, **kwargs)
    backward_time = backward(loss) if split == 'train' else 0
    if o.s == 1:
        print('Runtime: %.3fs' % (forward_time + backward_time))
    return loss.data[0]


# The test function
def run_test_epoch():
    torch.save(net.states, result_file_header + 'tmp.pt')
    net.reset_states()
    net.eval()
    val_loss_sum = 0
    for batch_id in range(0, o.test_batch_num):
        o.batch_id = batch_id
        loss = run_batch(batch_id, 'test')
        val_loss_sum = val_loss_sum + loss
        print('Validation %d / %d, loss = %.3f'% (batch_id+1, o.test_batch_num, loss))
    val_loss = val_loss_sum / o.test_batch_num
    print('Final validation loss: %.3f'% (val_loss))
    net.states = torch.load(result_file_header + 'tmp.pt')
    return val_loss


# The training function
i = benchmark['i_start']
iter_num = o.train_batch_num * o.epoch_num
train_loss_sum = 0
def run_train_epoch(batch_id_start):
    net.reset_states()
    net.train()
    global i, train_loss_sum
    for batch_id in range(batch_id_start, o.train_batch_num):
        o.batch_id = batch_id
        loss = run_batch(batch_id, 'train')
        train_loss_sum = train_loss_sum + loss
        i = i + 1
        if ((0 < o.reset_interval < 1 and np.random.random_sample() < o.reset_interval) or
            (o.reset_interval > 1 and i % round(max(o.reset_interval/o.T, 1)) == 0)):
            print('---------- State Reset ----------')
            net.reset_states()
        if o.print_interval > 0 and i % o.print_interval == 0:
            print('Epoch: %.2f/%d, iter: %d/%d, batch: %d/%d, loss: %.3f'% 
                  (i/o.train_batch_num, o.epoch_num, i, iter_num, batch_id+1, o.train_batch_num, loss))
        if o.train_log_interval > 0 and i % o.train_log_interval == 0:
            benchmark['train_loss'].append((i, train_loss_sum/o.train_log_interval))
            train_loss_sum = 0
        if o.validate_interval > 0 and (i % o.validate_interval == 0 or i == iter_num):
            val_loss = run_test_epoch() if o.test_batch_num > 0 else 0
            net.train()
            benchmark['val_loss'].append((i, val_loss))
            benchmark['i_start'] = i
            savepoint = {'o': vars(o), 'benchmark': benchmark}
            utils.save_json(savepoint, result_file_header + str(i) + '.json')
            savepoint['net_states'] = net.state_dict()
            # savepoint['optim_states'] = optimizer.state_dict()
            torch.save(savepoint, result_file_header + str(i) + '.pt')
        if o.save_interval > 0 and (i % o.save_interval == 0 or i == iter_num):
            benchmark['i_start'] = i
            savepoint = {'o': vars(o), 'benchmark': benchmark}
            utils.save_json(savepoint, result_file_header + 'latest.json')
            savepoint['net_states'] = net.state_dict()
            # savepoint['optim_states'] = optimizer.state_dict()
            torch.save(savepoint, result_file_header + 'latest.pt')
        print('-' * 80)


# Run the model
if o.train == 1:
    epoch_id_start = i // o.train_batch_num
    batch_id_start = i % o.train_batch_num
    for epoch_id in range(epoch_id_start, o.epoch_num):
        run_train_epoch(batch_id_start)
        batch_id_start = 0
else:
    run_test_epoch()