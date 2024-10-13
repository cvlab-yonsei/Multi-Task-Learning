import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.weighting.abstract_weighting import AbsWeighting

class DWA(AbsWeighting):

    def __init__(self):
        super(DWA, self).__init__()
        
    def backward(self, losses, **kwargs):
        T = kwargs['T']
        if self.epoch > 1:
            w_i = torch.Tensor(self.train_loss_buffer[:,self.epoch-1]/self.train_loss_buffer[:,self.epoch-2]).to(self.device)
            batch_weight = self.task_num*F.softmax(w_i/T, dim=-1)
        else:
            batch_weight = torch.ones_like(losses).to(self.device)
        loss = torch.mul(losses, batch_weight).sum()
        loss.backward()
        return batch_weight.detach().cpu().numpy()