import numpy as np
import torch
from torch.autograd import Function



class CheckBP(Function):

    def __init__(self, label, show):
        super(CheckBP, self).__init__()
        self.label = label
        self.show = show

    def forward(self, input):
        # print(self.label + ': forward passed.')
        # print(input)
        return input.clone()

    def backward(self, grad_output):
        grad_mean = grad_output.abs().mean()
        if self.show == 1:
            print('grad_' + self.label + ': ' + str(grad_mean))

        # assert grad_mean < 10, 'Abnormal gradients for ' + self.label + ': ' + str(grad_mean)
        # if grad_mean > 10:
        #     print('grad_' + self.label + ' is LARGE: ' + str(grad_mean))
        # if grad_mean != grad_mean:
        #     print('grad_' + self.label + ' is NAN!')
        return grad_output


class Identity(Function):

    def forward(self, input):
        return input

    def backward(self, grad_output):
        return grad_output


class Round(Function):

    def forward(self, input):
        return input.round()

    def backward(self, grad_output):
        return grad_output


class StraightThrough(Function):

    def forward(self, input):
        u = torch.rand(input.size()).cuda()
        return u.lt(input).type_as(input)

    def backward(self, grad_output):
        return grad_output


class ArgMax(Function):

    def forward(self, input):
        _, max_index = input.max(1) # N
        output = input.clone().zero_() # N * K
        return output.scatter_(1, max_index.unsqueeze(1), 1)

    def backward(self, grad_output):
        return grad_output


class PermutationMatrixCalculator(Function):

    def __init__(self, descend=True):
        super(PermutationMatrixCalculator, self).__init__()
        self.descend = descend

    def forward(self, input):
        _, sort_index = input.sort(1, self.descend) # N * K, sort the input along its last dimension
        output = input.new(input.size(0), input.size(1), input.size(1)).zero_() # N * K * K
        return output.scatter_(2, sort_index.unsqueeze(2), 1)

    def backward(self, grad_output):
        return grad_output.new(grad_output.size(0), grad_output.size(1)).zero_()