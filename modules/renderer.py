import os
import os.path as path
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import modules.submodules as smd
import modules.utils as utils


class Renderer(nn.Module):

    def __init__(self, o):
        super(Renderer, self).__init__()
        self.o = o
        self.i = 0 # a counter

    def forward(self, y_e, y_l, y_p, Y_s, Y_a, **kwargs):
        o = self.o
        Y_b = kwargs['Y_b'].view(-1, o.D, o.H, o.W) if 'Y_b' in kwargs.keys() else None
        if o.task == 'mnist':
            Y_s = Variable(Y_s.data.clone().fill_(1))

        # Get sampling grid
        y_e = y_e.view(-1, o.dim_y_e, 1, 1) # NTO * dim_y_e * 1 * 1
        y_p = y_p.view(-1, o.dim_y_p) # NTO * dim_y_p
        grid, area = self.get_sampling_grid(y_e, y_p) # NTO * H * W * 2
        area = area.sum() / o.O

        # Spatial transform
        Y_s = Y_s.view(-1, 1, o.h, o.w) * y_e # NTO * 1 * h * w 
        Y_a = Y_a.view(-1, o.D, o.h, o.w) * Y_s # NTO * D * h * w
        X_s = nn.functional.grid_sample(Y_s, grid) # NTO * 1 * H * W
        X_a = nn.functional.grid_sample(Y_a, grid) # NTO * D * H * W

        # Permute, and generate layers
        X_s = X_s.view(-1, o.O, 1 * o.H * o.W) # NT * O * 1HW
        X_a = X_a.view(-1, o.O, o.D * o.H * o.W) # NT * O * DHW
        y_l = y_l.view(-1, o.O, o.dim_y_l) # NT * O * dim_y_l
        y_l = y_l.transpose(1, 2) # NT * dim_y_l * O
        X_s = y_l.bmm(X_s).clamp(max=1) # NT * dim_y_l * 1HW
        X_a = y_l.bmm(X_a) # NT * dim_y_l * DHW
        if o.task == 'mnist':
            X_a = X_a.clamp(max=1)

        # Reconstruct iteratively
        X_s_split = torch.unbind(X_s.view(-1, o.dim_y_l, 1, o.H, o.W), 1)  # NT * 1 * H * W
        X_a_split = torch.unbind(X_a.view(-1, o.dim_y_l, o.D, o.H, o.W), 1) # NT * D * H * W
        X_r = Y_b if Y_b is not None else Variable(X_a_split[0].data.clone().zero_()) # NT * D * H * W
        for i in range(0, o.dim_y_l):
            # X_r = X_r + X_s_split[i] * (X_a_split[i] - X_r)
            X_r = X_r * (1 - X_s_split[i]) + X_a_split[i]
        X_r = X_r.view(-1, o.T, o.D, o.H, o.W) # N * T * D * H * W

        return X_r, area


    def get_sampling_grid(self, y_e, y_p):
        """
        y_e: N * dim_y_e * 1 * 1
        y_p: N * dim_y_p (scale_x, scale_y, trans_x, trans_y)
        """
        o = self.o

        # Generate 2D transformation matrix
        scale, ratio, trans_x, trans_y = y_p.split(1, 1) # N * 1
        scale = 1 + o.zeta_s*scale
        ratio = o.zeta_r[0] + o.zeta_r[1]*ratio
        ratio_sqrt = ratio.sqrt()
        area = 1 / (scale * scale)
        h_new = o.h * scale * ratio_sqrt
        w_new = o.w * scale / ratio_sqrt
        scale_x = o.W / w_new
        scale_y = o.H / h_new
        if o.bg == 0:
            trans_x = (1 - (o.w*2/3)/o.W) * trans_x
            trans_y = (1 - (o.h*2/3)/o.H) * trans_y
        zero = Variable(trans_x.data.clone().zero_()) # N * 1
        trans_mat = torch.cat((scale_x, zero, scale_x * trans_x, zero, scale_y, 
                               scale_y * trans_y), 1).view(-1, 2, 3) # N * 2 * 3

        # Convert to bounding boxes and save
        if o.metric == 1 and o.v == 0:
            bb_conf = y_e.data.view(-1, o.dim_y_e)
            bb_h = h_new.data
            bb_w = w_new.data
            bb_center_y = (-trans_y.data + 1)/2 * (o.H - 1) + 1 # [1, H]
            bb_center_x = (-trans_x.data + 1)/2 * (o.W - 1) + 1 # [1, W]
            bb_top = bb_center_y - (bb_h-1)/2
            bb_left = bb_center_x - (bb_w-1)/2
            bb = torch.cat((bb_left, bb_top, bb_w, bb_h, bb_conf), dim=1) # NTO * 5
            torch.save(bb.view(-1, o.T, o.O, 5), path.join(o.result_metric_dir, str(self.i)+'.pt'))
            self.i += 1

        # Generate sampling grid
        grid = nn.functional.affine_grid(trans_mat, torch.Size((trans_mat.size(0), o.D, o.H, o.W))) # N * H * W * 2
        return grid, area