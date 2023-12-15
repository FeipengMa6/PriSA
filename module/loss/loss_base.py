import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
EPISILON=1e-10
class NCELoss(torch.nn.Module):
    def __init__(self, temperature=1,d=0.1):
        super(NCELoss, self).__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=1)
        self.d = d
    def where(self, cond, x_1, x_2):
        cond = cond.type(torch.float32)
        return (cond * x_1) + ((1 - cond) * x_2)
    def forward(self, f1, f2, targets):
        f1 = F.normalize(f1, dim=1)
        f2 = F.normalize(f2, dim=1)
        if(self.d > 0):
            distance_targets = abs(targets.unsqueeze(1) - targets)
            mask_distance = (distance_targets>self.d).bool()
            polarity_targets = (targets>0).int()
            mask_polarity = abs(polarity_targets.unsqueeze(1)-polarity_targets).bool()
            self_mask = (mask_distance + mask_polarity).int()
        else:
            mask = targets.unsqueeze(1) - targets
            self_mask = (torch.zeros_like(mask) != mask).int() # remove negative term
        dist = (f1.unsqueeze(1) - f2).pow(2).sum(2)
        cos = 1 - 0.5 * dist
        pred_softmax = F.softmax(cos / self.temperature,dim=1) 
        log_pos_softmax = - torch.log(pred_softmax + EPISILON) * (1 - self_mask.float())
        log_neg_softmax = - torch.log(1 - pred_softmax + EPISILON) * self_mask.float()
        log_softmax = log_pos_softmax.sum(1) / (1 - self_mask).sum(1).float() + log_neg_softmax.sum(1) / self_mask.sum(1).float()
        loss = log_softmax
        return loss.mean()