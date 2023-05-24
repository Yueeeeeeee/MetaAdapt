import logging
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class LSLRGradientDescentLearningRule(nn.Module):
    def __init__(self, device, total_num_inner_loop_steps, init_learning_rate=1e-5, use_learnable_lr=True, lr_of_lr=1e-3):
        super(LSLRGradientDescentLearningRule, self).__init__()
        assert init_learning_rate > 0., 'learning_rate should be positive.'

        self.init_learning_rate = torch.ones(1) * init_learning_rate
        self.init_learning_rate.to(device)
        self.total_num_inner_loop_steps = total_num_inner_loop_steps
        self.use_learnable_lr = use_learnable_lr
        self.lr_of_lr = lr_of_lr

    def initialize(self, names_weights_dict):
        self.names_learning_rates_dict = nn.ParameterDict()
        for key, param in names_weights_dict:
            self.names_learning_rates_dict[key.replace(".", "-")] = nn.Parameter(
                data=torch.ones(self.total_num_inner_loop_steps) * self.init_learning_rate,
                requires_grad=self.use_learnable_lr)
    
    def update_lrs(self, loss, scaler=None):
        if self.use_learnable_lr:
            if scaler is not None:
                scaled_grads = torch.autograd.grad(scaler.scale(loss), self.names_learning_rates_dict.values())
                inv_scale = 1. / scaler.get_scale()
                grads = [p * inv_scale for p in scaled_grads]
                if any([False in torch.isfinite(g) for g in grads]):
                    print('Invalid LR gradients, adjust scale and zero out gradients')
                    if scaler.get_scale() * scaler.get_backoff_factor() >= 1.:
                        scaler.update(scaler.get_scale() * scaler.get_backoff_factor())
                    for g in grads: g.zero_()
            else:
                grads = torch.autograd.grad(loss, self.names_learning_rates_dict.values())
                if any([False in torch.isfinite(g) for g in grads]):
                    print('Invalid LR gradients, zero out gradients')
                    for g in grads: g.zero_()
            
            for idx, key in enumerate(self.names_learning_rates_dict.keys()):
                self.names_learning_rates_dict[key] = nn.Parameter(self.names_learning_rates_dict[key] - self.lr_of_lr * grads[idx])
    
    def update_params(self, names_weights_dict, grads, num_step):
        return OrderedDict(
            (key, names_weights_dict[key] - self.names_learning_rates_dict[key.replace(".", "-")][num_step] * grads[idx])
            for idx, key in enumerate(names_weights_dict.keys()))