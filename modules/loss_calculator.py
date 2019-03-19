import torch
import torch.nn as nn
from torch.autograd import Variable
import modules.submodules as smd
import modules.utils as utils



class LossCalculator(nn.Module):


    def __init__(self, o):
        super(LossCalculator, self).__init__()
        self.o = o
        self.log = smd.Log()
        self.mse = nn.MSELoss(size_average=False)
        self.mmd = smd.MMDLoss(sigmas=[0.01, 0.03, 0.1, 0.3, 1, 3])
        # self.bce_loss = nn.BCELoss(size_average=False)


    def forward(self, output, target, area, **kwargs):
        o = self.o
        losses = {}

        # Reconstruction loss
        loss_recon = self.mse(output, target)
        loss = loss_recon
        losses['recon'] = loss_recon.data[0]

        # Tightness loss
        lam_t = 0.1 if o.task == 'duke' else 13
        loss_tight = lam_t * area
        loss = loss + loss_tight
        losses['tight'] = loss_tight.data[0]

        # Entropy loss of y_e
        y_e = kwargs['y_e']  # N * T * O * 1
        lam_e = 1
        loss_entr = lam_e * self.calc_entropy(y_e)
        loss = loss + loss_entr
        losses['entr'] = loss_entr.data[0]

        # Appearance loss of Y_a
        if 'Y_a' in kwargs.keys():
            lam_a = 0.1
            app_sum_thresh = 20
            Y_a = kwargs['Y_a']  # N * T * O * D * h * w
            Y_a_sum_indiv = Y_a.view(-1, o.T, o.O, o.D*o.h*o.w).sum(3)  # N * T * O
            loss_app_indiv = (app_sum_thresh - Y_a_sum_indiv).clamp(min=0)  # N * T * O
            loss_app = lam_a * loss_app_indiv.sum()  # 1
            loss = loss + loss_app
            losses['loss_app'] = loss_app.data[0]

        if torch.cuda.current_device() == 0:
            msg = ""
            for k in losses.keys():
                msg = msg + k + ": %.3f, "
                losses[k] /= loss.data[0]
            print(msg[:-2] % tuple(losses.values()))
        return loss


    def calc_entropy(self, x):
        log = self.log
        x_not = 1 - x
        return -(x * log(x) + x_not * log(x_not)).sum()