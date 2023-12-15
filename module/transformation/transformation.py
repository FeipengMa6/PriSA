import torch
import torch.nn as nn 
import torch.nn.functional as F
import random
import numpy as np
from torch.nn.utils.rnn import pad_sequence
class Augmentation(object):
    def __init__(self,aug='crop',crop_scale=0.08,cutout_scale=0.2,reverse_p=0.2,stride=3,multiscale_p=0.1,disorder_p=0.1):
        self.aug = aug
        self.crop_scale = crop_scale
        self.cutout_scale = cutout_scale
        self.reverse_p = reverse_p
        self.stride = stride
        self.multiscale_p = multiscale_p
        self.disorder_p = disorder_p
        self.length = None
    def run(self,x,length=None):
        with torch.no_grad():
            aug_list = self.aug.split('_')
            for aug_type in aug_list:
                x = getattr(self,aug_type)(x,length)
            return x
    def crop(self,x,length):
        B,_,_ = x.shape
        length = length.cpu()
        x_ = x[:,1:,:] 
        x_list= []
        scale=[self.crop_scale,1.0]
        rate = np.random.uniform(scale[0],scale[1])
        for i in range(B):
            sample_len = int(length[i] * rate)
            begin_pos = np.random.randint(low=0,high=length[i]-sample_len+1)
            x_list.append(x_[i,begin_pos:begin_pos+sample_len,:])
        x_ = pad_sequence(x_list,batch_first=True,padding_value=0.0)
        return torch.cat([x[:,0,:].unsqueeze(1),x_],dim=1)
    def cutout(self,x,length):
        B,_,_ = x.shape
        length = length.cpu()
        x_ = x.clone() 
        scale=(0,self.cutout_scale)
        rate = np.random.uniform(scale[0],scale[1])
        for i in range(B):
            sample_len = int(length[i] * rate)
            begin_pos = np.random.randint(low=1,high=length[i]-sample_len+2)
            x_[i,begin_pos:begin_pos+sample_len,:] = 0.0
        return x_
    def reverse(self,x,length):
        prob = np.random.uniform(low=0,high=1)
        length = length.cpu()
        if(prob>self.reverse_p):
            return x
        B,_,_ = x.shape
        x_ = x[:,1:,:] 
        x_list = []
        for i in range(B):
            x_flip = torch.flip(x_[i,:length[i],:],dims=[1])
            x_list.append(x_flip)
        x_ = pad_sequence(x_list,batch_first=True,padding_value=0.0)
        return torch.cat([x[:,0,:].unsqueeze(1),x_],dim=1)
    def multiscale(self,x,length):
        prob = np.random.uniform(low=0,high=1)
        if(prob>self.multiscale_p):
            return x
        x_ = x[:,1:,:]
        x_ = F.avg_pool1d(x_.transpose(1,2),kernel_size=self.stride, stride=self.stride,padding=0).transpose(1,2)
        return torch.cat([x[:,0,:].unsqueeze(1),x_],dim=1)
    def disorder(self,x,length):
        x_s = torch.sum(x[:,1:,:],dim=-1) 
        x_o = torch.zeros_like(x[:,1:,:])
        for b in range(x.shape[0]):
            sample = x_s[b,:] 
            ind = torch.nonzero(sample)
            if(len(ind)==0):
                continue
            else:
                ind = ind.squeeze(1)
                random.shuffle(ind)
                x_o[b,:len(ind),:] = x[b,ind,:]
        return torch.cat([x[:,0,:].unsqueeze(1),x_o],dim=1)
    def sample(self,x,length):
        B,_,_ = x.shape
        x_ = x[:,1:,:]
        scale=(0.08,1.0)
        rate = np.random.uniform(scale[0],scale[1])
        x_list = []
        for i in range(B):
            sample_len = int(length[i] * rate)
            index = random.sample(range(length[i]),sample_len)
            index.sort()
            x_list.append(x_[i,index,:])
        x_ = pad_sequence(x_list,batch_first=True,padding_value=0.0)
        return torch.cat([x[:,0,:].unsqueeze(1),x_],dim=1)
    def resize(self,x,size):
        pass
