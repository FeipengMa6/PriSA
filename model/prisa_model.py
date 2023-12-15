import torch
import torch.nn as nn
import sys
import random
from src.f_transformer import TransformerEncoderLayer
from src.position_embedding import SinusoidalPositionalEmbedding
from src.bert import BERT
from module import task_head
from module.transformation.transformation import Augmentation
from src.f_transformer import padding_mask,text_padding_mask
import math
import torch.nn.functional as F
import random
def calculate_len(x):
    with torch.no_grad():
        x_len = torch.sum(x,dim=-1)
        x_len = (x_len!=0).int() 
        x_len = x_len.sum(dim=-1)
    return x_len
def mixup(x_v,x_a,percent):
    seq_len = x_v.shape[1]
    sample_num = int(seq_len*percent)
    index = random.sample(range(seq_len),sample_num)
    index.sort()
    temp = x_v[:,index,:].clone()
    x_v[:,index,:] = x_a[:,index,:].clone()
    x_a[:,index,:] = temp
    return x_v,x_a
class prisa_model(nn.Module):
    def __init__(self,cfg):
        super(prisa_model,self).__init__()
        self.cfg = cfg
        self.dropout = self.cfg.dout_p 
        self.embed_scale = math.sqrt(self.cfg.embed_dim)
        self.embed_positions = SinusoidalPositionalEmbedding(self.cfg.embed_dim)
        self.attn_mask = self.cfg.attn_mask
        self.augmentation = Augmentation(self.cfg.aug,self.cfg.crop_scale,
                                        self.cfg.cutout_scale,self.cfg.reverse_p,
                                        self.cfg.stride,self.cfg.multiscale_p,
                                        self.cfg.disorder_p)
        if(self.cfg.using_bert):
            self.t_model = BERT(bert_path=self.cfg.bert_path)
        self.proj_t = nn.Conv1d(self.cfg.t_in_dim,self.cfg.t_hid_dim,kernel_size=1,padding=0,bias=False)
        self.proj_v = nn.Conv1d(self.cfg.v_in_dim,self.cfg.v_hid_dim,kernel_size=1,padding=0,bias=False)
        self.proj_a = nn.Conv1d(self.cfg.a_in_dim,self.cfg.a_hid_dim,kernel_size=1,padding=0,bias=False)
        self.mlp_a = nn.Sequential(nn.Linear(self.cfg.a_hid_dim,self.cfg.a_hid_dim), nn.ReLU(),nn.Linear(self.cfg.a_hid_dim,self.cfg.a_hid_dim))
        self.mlp_v = nn.Sequential(nn.Linear(self.cfg.v_hid_dim,self.cfg.v_hid_dim), nn.ReLU(),nn.Linear(self.cfg.v_hid_dim,self.cfg.v_hid_dim))
        self.at_layers = nn.ModuleList([])
        self.vt_layers = nn.ModuleList([])
        self.a_layers = nn.ModuleList([])
        self.v_layers = nn.ModuleList([])
        for layer in range(self.cfg.private_layers):
            new_layer_v = TransformerEncoderLayer(self.cfg.embed_dim,
                                                num_heads=self.cfg.num_heads,
                                                attn_dropout=self.cfg.attn_dropout_v,
                                                relu_dropout=self.cfg.relu_dropout,
                                                res_dropout=self.cfg.res_dropout)
            new_layer_a = TransformerEncoderLayer(self.cfg.embed_dim,
                                                num_heads=self.cfg.num_heads,
                                                attn_dropout=self.cfg.attn_dropout_a,
                                                relu_dropout=self.cfg.relu_dropout,
                                                res_dropout=self.cfg.res_dropout)
            self.v_layers.append(new_layer_v)
            self.a_layers.append(new_layer_a)
        for layer in range(self.cfg.layer+self.cfg.num_layers_sa):
            new_layer = TransformerEncoderLayer(self.cfg.embed_dim,
                                                num_heads=self.cfg.num_heads,
                                                attn_dropout=self.cfg.attn_dropout_a,
                                                relu_dropout=self.cfg.relu_dropout,
                                                res_dropout=self.cfg.res_dropout)
            self.at_layers.append(new_layer)
        for layer in range(self.cfg.layer+self.cfg.num_layers_sa):
            new_layer = TransformerEncoderLayer(self.cfg.embed_dim,
                                                num_heads=self.cfg.num_heads,
                                                attn_dropout=self.cfg.attn_dropout_v,
                                                relu_dropout=self.cfg.relu_dropout,
                                                res_dropout=self.cfg.res_dropout)
            self.vt_layers.append(new_layer)
        self.layernorm_a = nn.ModuleList([nn.LayerNorm(self.cfg.embed_dim) for _ in range(2)])
        self.layernorm_v = nn.ModuleList([nn.LayerNorm(self.cfg.embed_dim) for _ in range(2)])
        self.f_task_head = task_head.get_instance(self.cfg.task_head_type,self.cfg,modality='fusion')
        self.a_task_head = task_head.get_instance(self.cfg.task_head_type,self.cfg,modality='audio')
        self.v_task_head = task_head.get_instance(self.cfg.task_head_type,self.cfg,modality='video')
        self.ap_task_head = task_head.get_instance(self.cfg.task_head_type,self.cfg,modality='audio')
        self.vp_task_head = task_head.get_instance(self.cfg.task_head_type,self.cfg,modality='video')
        self.t_task_head = task_head.get_instance(self.cfg.task_head_type,self.cfg,modality='video')
    def forward(self,input_dict):
        x_v = input_dict['vision']
        x_a = input_dict['audio']
        x_t = input_dict['text_emb']
        x_t_id = input_dict['text_id']
        x_t_mask = input_dict['text_mask']
        vat_len = input_dict['len']
        B = x_v.shape[0]
        x_v_cls = torch.zeros([B,1,self.cfg.v_in_dim]).cuda()
        x_a_cls = torch.zeros([B,1,self.cfg.a_in_dim]).cuda()
        x_v = torch.cat([x_v_cls,x_v[:,:-1,:]],dim=1)
        x_a = torch.cat([x_a_cls,x_a[:,:-1,:]],dim=1)
        B,v_len,_ = x_v.shape
        _,a_len,_ = x_a.shape
        _,t_len,_ = x_t.shape
        v_mask = None
        a_mask = None
        if(self.cfg.using_bert):
            x_t = self.t_model(x_t_id,x_t_mask)
        x_a = self.proj_a(x_a.permute(0,2,1)).permute(0,2,1)
        x_v = self.proj_v(x_v.permute(0,2,1)).permute(0,2,1)
        x_t = self.proj_t(x_t.permute(0,2,1)).permute(0,2,1)
        if(self.attn_mask):
            v_mask = padding_mask(B,v_len,vat_len[:,0])
            a_mask = padding_mask(B,a_len,vat_len[:,1])
        x_v_p = x_v + self.embed_positions(x_v[:, :, 0])
        x_a_p = x_a + self.embed_positions(x_a[:, :, 0])
        for i in range(self.cfg.private_layers):    
            x_a_private = self.a_layers[i](x=x_a_p.transpose(0,1),mask=a_mask).transpose(0,1)
            x_v_private = self.v_layers[i](x=x_v_p.transpose(0,1),mask=v_mask).transpose(0,1)
        x_a_private = self.layernorm_a[0](x_a_private)
        x_v_private = self.layernorm_v[0](x_v_private)
        x_a_private = x_a_private[:,0,:]
        x_v_private = x_v_private[:,0,:]
        x_t_private = x_t[:,0,:]
        if(self.training and self.cfg.contrastive_learning):
            x_v = self.augmentation.run(x_v,vat_len[:,0])
            x_a = self.augmentation.run(x_a,vat_len[:,1])
            with torch.no_grad():
                vat_len[:,0] = calculate_len(x_v)
                vat_len[:,1] = calculate_len(x_a) 
        x_a = self.embed_scale * x_a
        x_v = self.embed_scale * x_v
        x_t = self.embed_scale * x_t
        x_v += self.embed_positions(x_v[:, :, 0])
        x_a += self.embed_positions(x_a[:, :, 0])
        x_v = F.dropout(x_v,p=self.dropout,training=self.training)
        x_a = F.dropout(x_a,p=self.dropout,training=self.training)
        x_t = F.dropout(x_t,p=self.dropout,training=self.training)
        if(self.attn_mask):
            v_mask = text_padding_mask(x_t_mask)
            a_mask = text_padding_mask(x_t_mask)
        for i in range(self.cfg.layer):
            x_a = self.at_layers[i](x=x_a.transpose(0,1),x_k=x_t.transpose(0,1),x_v=x_t.transpose(0,1),mask=a_mask).transpose(0,1)
            x_v = self.vt_layers[i](x=x_v.transpose(0,1),x_k=x_t.transpose(0,1),x_v=x_t.transpose(0,1),mask=v_mask).transpose(0,1)
        begin = self.cfg.layer
        if(self.attn_mask):
            B,v_len,_ = x_v.shape
            _,a_len,_ = x_a.shape
            v_mask = padding_mask(B,v_len,vat_len[:,0])
            a_mask = padding_mask(B,a_len,vat_len[:,1])
        for i in range(begin,begin+self.cfg.num_layers_sa):
            x_a = self.at_layers[i](x=x_a.transpose(0,1),mask=a_mask).transpose(0,1)
            x_v = self.vt_layers[i](x=x_v.transpose(0,1),mask=v_mask).transpose(0,1)
        x_a = self.layernorm_a[1](x_a)
        x_v = self.layernorm_v[1](x_v)
        v_last_state = x_v[:,0,:]
        a_last_state = x_a[:,0,:]
        fusion_vat = torch.cat([v_last_state,a_last_state,x_v_private,x_a_private],dim=-1)
        state = {}
        state['fusion'] = fusion_vat
        state['v_private'] = x_v_private
        state['a_private'] = x_a_private
        state['t_private'] = x_t_private
        state['vt_fusion'] = v_last_state
        state['at_fusion'] = a_last_state
        score = {}
        score['fusion'] = self.f_task_head(fusion_vat)
        score['audio'] = self.a_task_head(a_last_state)
        score['video'] = self.v_task_head(v_last_state)
        score['audio_private'] = self.ap_task_head(x_a_private)
        score['video_private'] = self.vp_task_head(x_v_private)
        score['text'] = self.t_task_head(x_t_private)
        return score,state
