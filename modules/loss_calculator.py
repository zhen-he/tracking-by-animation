import torch
import torch.nn as nn
import modules.submodules as smd
import modules.utils as utils



class LossCalculator(nn.Module):


    def __init__(self, o):
        super(LossCalculator, self).__init__()
        self.o = o
        self.log = smd.Log()
        self.mse = nn.MSELoss(reduction='sum')
        # self.bce_loss = nn.BCELoss(size_average=False)


    def forward(self, output, target, area, **kwargs):
        o = self.o
        losses = {}

        # Reconstruction loss
        if 'Y_b' in kwargs.keys():
            Y_b = kwargs['Y_b']
            loss_recon = self.mse(output, target) + self.mse((output - Y_b).abs(), (target - Y_b).abs())
        else:
            loss_recon = self.mse(output, target)
        loss = loss_recon
        losses['recon'] = loss_recon.item()

        # Tightness loss
        lam_t = 0.1 if o.task == 'duke' else 13
        loss_tight = lam_t * area.sum()
        loss = loss + loss_tight
        losses['tight'] = loss_tight.item()

        # Entropy loss of y_e
        y_e = kwargs['y_e']  # N * T * O * 1
        lam_e = 1
        loss_entr = lam_e * self.calc_entropy(y_e)
        loss = loss + loss_entr
        losses['entr'] = loss_entr.item()

        # Appearance loss of Y_a
        if 'Y_a' in kwargs.keys():
            lam_a = 0.1
            app_sum_thresh = 20
            Y_a = kwargs['Y_a']  # N * T * O * D * h * w
            Y_a_sum_indiv = Y_a.view(-1, o.T, o.O, o.D*o.h*o.w).sum(3)  # N * T * O
            loss_app_indiv = (app_sum_thresh - Y_a_sum_indiv).clamp(min=0)  # N * T * O
            loss_app = lam_a * loss_app_indiv.sum()  # 1
            loss = loss + loss_app
            losses['loss_app'] = loss_app.item()

        if torch.cuda.current_device() == 0:
            msg = ""
            for k in losses.keys():
                msg = msg + k + ": %.3f, "
                losses[k] /= loss.item()
            print(msg[:-2] % tuple(losses.values()))
        return loss


    def calc_entropy(self, x):
        log = self.log
        x_not = 1 - x
        return -(x * log(x) + x_not * log(x_not)).sum()
