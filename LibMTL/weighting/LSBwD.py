import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.weighting.abstract_weighting import AbsWeighting

class LSBwD(AbsWeighting):
    r"""Loss Scale Balancing (LSB).
    
    """
    def __init__(self):
        super(LSBwD, self).__init__()
        
    def backward(self, losses, **kwargs):
        if not hasattr(self, 'prev_weight'):
            self.prev_weight = torch.ones_like(losses).detach() / self.task_num
            self.loss_cache = 0
            self.losses_cache = 0
            self.iter = 0
            self.beta = - (1.0 / self.epochs)
        
        if not hasattr(self, 'prev2_weight'):
            loss = torch.mul(losses, self.prev_weight).sum()
        else:
            
            loss = torch.mul(losses, self.alpha * self.difficulties * self.prev_weight).sum()
        
        self.loss_cache += loss.detach() / self.train_batch
        self.losses_cache += losses.detach() / self.train_batch
        self.iter += 1
        if (self.iter+1) % self.train_batch==0: # epoch == period
            if (self.iter+1) > self.train_batch:
                self.prev2_weight = self.prev_weight
                temp_prev_weight = self.loss_cache / (self.losses_cache * self.task_num)
                self.beta += 1.0 / self.epochs
                self.difficulties = torch.pow((temp_prev_weight/self.prev2_weight) / (self.losses_cache/self.losses_cache_prev2), self.beta)
                self.alpha = self.task_num / sum(self.difficulties)
            self.prev_weight = self.loss_cache / (self.losses_cache * self.task_num)
            self.losses_cache_prev2 = temp_prev_weight if (self.iter+1) > self.train_batch else self.losses_cache
            self.loss_cache = 0
            self.losses_cache = 0
            
        loss.backward()
        # return self.prev_weight.detach().cpu().numpy()