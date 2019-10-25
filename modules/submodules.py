import numpy as np
import torch
import torch.nn as nn
import functions.submodules as F




def norm_grad(input, max_norm):
    if input.requires_grad:
        def norm_hook(grad):
            N = grad.size(0) # batch number
            norm = grad.view(N, -1).norm(p=2, dim=1) + 1e-6
            scale = (norm / max_norm).clamp(min=1).view([N]+[1]*(grad.dim()-1))
            return grad / scale

            # clip_coef = float(max_norm) / (grad.norm(2).data[0] + 1e-6)
            # return grad.mul(clip_coef) if clip_coef < 1 else grad
        input.register_hook(norm_hook)


def clip_grad(input, value):
    if  input.requires_grad:
        input.register_hook(lambda g: g.clamp(-value, value))


def scale_grad(input, scale):
    if input.requires_grad:
        input.register_hook(lambda g: g * scale)


def func(func_name):
    if func_name is None:
        return None
    elif func_name == 'tanh':
        return nn.Tanh()
    elif func_name == 'relu':
        return nn.ReLU()
    elif func_name == 'sigmoid':
        return nn.Sigmoid()
    elif func_name == 'softmax':
        return nn.Softmax(dim=1)
    else:
        assert False, 'Invalid func_name.'


class CheckBP(nn.Module):

    def __init__(self, label='a', show=1):
        super(CheckBP, self).__init__()
        self.label = label
        self.show = show

    def forward(self, input):
        return F.CheckBP.apply(input, self.label, self.show)


class Identity(nn.Module):

    def forward(self, input):
        return F.Identity.apply(input)


class Log(nn.Module):

    def __init__(self, eps=1e-20):
        super(Log, self).__init__()
        self.eps = eps

    def forward(self, input):
        return (input + self.eps).log()


class Round(nn.Module):
    """
    The round operater which is similar to the deterministic Straight-Through Estimator
    It forwards by rounding the input, and backwards with the original output gradients
    """
    def forward(self, input):
        return F.Round.apply(input)


class StraightThrough(nn.Module):
    """
    The stochastic Straight-Through Estimator
    It forwards by sampling from the input probablilities, and backwards with the original output gradients
    """
    def forward(self, input):
        return F.StraightThrough.apply(input)


class ArgMax(nn.Module):
    """
    Input: N * K matrix, where N is the batch size
    Output: N * K matrix, the one-hot encoding of arg_max(input) along the last dimension
    """
    def forward(self, input):
        assert input.dim() == 2, 'only support 2D arg max'
        return F.ArgMax.apply(input)


class STGumbelSigmoid(nn.Module):

    def __init__(self, tao=1.0):
        super(STGumbelSigmoid, self).__init__()
        self.tao = tao
        self.log = Log()
        self.round = Round()

    def forward(self, mu):
        log = self.log
        u1 = torch.rand(mu.size()).cuda()
        u2 = torch.rand(mu.size()).cuda()
        a = (log(mu) - log(-log(u1)) - log(1 - mu) + log(-log(u2))) / self.tao
        return self.round(a.sigmoid())


class STGumbelSoftmax(nn.Module):

    def __init__(self, tao=1.0):
        super(STGumbelSoftmax, self).__init__()
        self.tao = tao
        self.log = Log()
        self.softmax = nn.Softmax(dim=1)
        self.arg_max = ArgMax()

    def forward(self, mu):
        log = self.log
        u = torch.rand(mu.size()).cuda() # N * K
        # mu = CheckBP('mu')(mu)
        a = (log(mu) - log(-log(u))) / self.tao
        # a = CheckBP('a')(a)
        return self.arg_max(self.softmax(a))


class GaussianSampler(nn.Module):

    def forward(self, mu, log_var):
        standard_normal = torch.randn(mu.size()).cuda()
        return mu + (log_var * 0.5).exp() * standard_normal


class PermutationMatrixCalculator(nn.Module):
    """
    Input: N * K matrix, where N is the batch size
    Output: N * K * K tensor, with each K * K matrix to sort the corresponding row of the input 
    """
    def __init__(self, descend=True):
        super(PermutationMatrixCalculator, self).__init__()
        self.descend = descend

    def forward(self, input):
        assert input.dim() == 2, 'only support 2D input'
        return F.PermutationMatrixCalculator.apply(input, self.descend)


class Conv(nn.Module):

    def __init__(self, conv_features, conv_kernels, out_sizes, bn=0, dp=0):
        super(Conv, self).__init__()
        self.layer_num = len(conv_features) - 1
        self.out_sizes = out_sizes
        assert self.layer_num == len(conv_kernels) == len(out_sizes) > 0, 'Invalid conv parameters'
        self.bn = bn
        self.dp = dp
        # Convolutional block
        for i in range(0, self.layer_num):
            setattr(self, 'conv'+str(i), nn.Conv2d(conv_features[i], conv_features[i+1],
                (conv_kernels[i][0], conv_kernels[i][1]), stride=1,
                padding=(conv_kernels[i][0]//2, conv_kernels[i][1]//2)))
            if bn == 1:
                setattr(self, 'bn'+str(i), nn.BatchNorm2d(conv_features[i+1]))
            setattr(self, 'pool'+str(i), nn.AdaptiveMaxPool2d(tuple(out_sizes[i])))
            if dp == 1:
                setattr(self, 'dp'+str(i), nn.Dropout2d(0.2))
        # Transformations
        self.tranform = func('relu')

    def forward(self, X):
        H = X # N * D * H * W
        for i in range(0, self.layer_num):
            H = getattr(self, 'conv'+str(i))(H)
            if self.bn == 1:
                H = getattr(self, 'bn'+str(i))(H)
            H = getattr(self, 'pool'+str(i))(H)
            if self.dp == 1:
                H = getattr(self, 'dp'+str(i))(H)
            # if i == self.layer_num - 1:
            #     print(H.data[0, :, H.size(2)//2, H.size(3)//2].reshape(1, -1))
            H = self.tranform(H)
            # if i == self.layer_num - 1:
            #     print(H.data[0, :, H.size(2)//2, H.size(3)//2].reshape(1, -1))
        return H


class DeConv(nn.Module):

    def __init__(self, scales, conv_features, conv_kernels, conv_paddings, out_trans=None, bn=0, dp=0):
        super(DeConv, self).__init__()
        self.layer_num = len(conv_features) - 1
        self.scales = scales
        assert self.layer_num == len(scales) == len(conv_kernels) == len(conv_paddings) > 0, \
            'Invalid deconv parameters'
        self.bn = bn
        self.dp = dp
        # Convolutional block
        for i in range(0, self.layer_num):
            if scales[i] > 1:
                setattr(self, 'unpool'+str(i), nn.Upsample(scale_factor=scales[i], mode='nearest'))
            setattr(self, 'conv'+str(i), nn.Conv2d(conv_features[i], conv_features[i+1], conv_kernels[i],
                                                   stride=1, padding=tuple(conv_paddings[i])))
            if bn == 1:
                setattr(self, 'bn'+str(i), nn.BatchNorm2d(conv_features[i+1]))
            if dp == 1:
                setattr(self, 'dp'+str(i), nn.Dropout2d(0.2))
        # Transformations
        self.transform = func('relu')
        self.out_trans_func = func(out_trans)

    def forward(self, X):
        H = X # N * D * H * W
        # Hidden layers
        for i in range(0, self.layer_num):
            if self.scales[i] > 1:
                H = getattr(self, 'unpool'+str(i))(H)
            H = getattr(self, 'conv'+str(i))(H)
            if self.bn == 1:
                H = getattr(self, 'bn'+str(i))(H)
            if self.dp == 1:
                H = getattr(self, 'dp'+str(i))(H)
            if i < self.layer_num - 1:
                H = self.transform(H)
        # Output layer
        if self.out_trans_func is not None:
            H = self.out_trans_func(H)
        return H


class FCN(nn.Module):

    def __init__(self, features, hid_trans='tanh', out_trans=None, hid_bn=0, out_bn=0):
        super(FCN, self).__init__()
        self.layer_num = len(features) - 1
        assert self.layer_num > 0, 'Invalid fc parameters'
        self.hid_bn = hid_bn
        self.out_bn = out_bn
        # Linear layers
        for i in range(0, self.layer_num):
            setattr(self, 'fc'+str(i), nn.Linear(features[i], features[i+1]))
            if hid_bn == 1:
                setattr(self, 'hid_bn_func'+str(i), nn.BatchNorm1d(features[i+1]))
        if out_bn == 1:
            self.out_bn_func = nn.BatchNorm1d(features[-1])
        # Transformations        
        self.hid_trans_func = func(hid_trans)
        self.out_trans_func = func(out_trans)

    def forward(self, X):
        H = X
        # Hidden layers
        for i in range(0, self.layer_num):
            H = getattr(self, 'fc'+str(i))(H)
            if i < self.layer_num - 1:
                if self.hid_bn == 1:
                    H = getattr(self, 'hid_bn_func'+str(i))(H)
                H = self.hid_trans_func(H)
        # Output layer
        if self.out_bn == 1:
            H = self.out_bn_func(H)
        if self.out_trans_func is not None:
            H = self.out_trans_func(H)
        return H


class CNN(nn.Module):

    def __init__(self, params):
        super(CNN, self).__init__()
        self.conv = Conv(params['conv_features'], params['conv_kernels'], params['out_sizes'], 
                         bn=params['bn'])
        self.fcn = FCN(params['fc_features'], hid_trans='relu', out_trans=params['out_trans'], 
                       hid_bn=params['bn'], out_bn=params['bn'])

    def forward(self, X):
        # X: N * D * H * W
        # Conv
        H = self.conv(X) # N * D_out1 * H_out1 * W_out1
        # H = CheckBP('H_Conv')(H)
        # FCN
        H = H.view(H.size(0), -1) # N * (D_out1 * H_out1 * W_out1)
        H = self.fcn(H) # N * D_out2
        return H


class DCN(nn.Module):

    def __init__(self, params):
        super(DCN, self).__init__()
        self.fcn = FCN(params['fc_features'], hid_trans='relu', out_trans='relu', hid_bn=params['bn'], 
                       out_bn=params['bn'])
        self.deconv = DeConv(params['scales'], params['conv_features'], params['conv_kernels'], 
                             params['conv_paddings'], out_trans=params['out_trans'], bn=params['bn'])
        self.H_in, self.W_in = params['H_in'], params['W_in']

    def forward(self, X):
        # X: N * D
        # FCN
        H = self.fcn(X) # N * (D_out1 * H_out1 * W_out1)
        # Deconv
        H = H.view(H.size(0), -1, self.H_in, self.W_in) # N * D_out1 * H_out1 * W_out1
        H = self.deconv(H) # N * D_out2 * H_out2 * W_out2
        return H
