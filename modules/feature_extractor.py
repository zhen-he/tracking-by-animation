import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import modules.submodules as smd
import modules.utils as utils


class FeatureExtractor(nn.Module):

    def __init__(self, o):
        super(FeatureExtractor, self).__init__()
        self.o = o
        params = o.cnn.copy()
        params['conv_features'] = [o.D+2] + params['conv_features']
        self.cnn = smd.Conv(params['conv_features'], params['conv_kernels'], params['out_sizes'], bn=params['bn'])

    def forward(self, X_seq):
        o = self.o
        X_seq = X_seq.view(-1, X_seq.size(2), X_seq.size(3), X_seq.size(4))  # NT * D+2 * H * W        
        C3_seq = self.cnn(X_seq)                                             # NT * C3_3 * C3_1 * C3_2
        C3_seq = C3_seq.permute(0, 2, 3, 1).contiguous()                     # NT * C3_1 * C3_2 * C3_3
        C2_seq = C3_seq.view(-1, o.T, o.dim_C2_1, o.dim_C2_2)                # N * T * C2_1 * C2_2
        return C2_seq