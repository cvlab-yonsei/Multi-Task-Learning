import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.weighting.abstract_weighting import AbsWeighting

class LSBwoD(AbsWeighting):
    r"""Loss Scale Balancing (LSB).
    
    """
    def __init__(self):
        super(LSBwoD, self).__init__()
        
    def backward(self, losses, **kwargs):
        if not hasattr(self, 'prev_weight'):
            self.prev_weight = torch.ones_like(losses).detach() / self.task_num
            self.loss_cache = 0
            self.losses_cache = 0
            self.iter = 0
        
        loss = torch.mul(losses, self.prev_weight).sum()
        self.loss_cache += loss.detach() / self.train_batch
        self.losses_cache += losses.detach() / self.train_batch
        self.iter += 1
        if (self.iter+1) % self.train_batch==0:
            self.prev_weight = self.loss_cache / (self.losses_cache * self.task_num)
            self.loss_cache = 0
            self.losses_cache = 0
            
        loss.backward()
        return self.prev_weight.detach().cpu().numpy()