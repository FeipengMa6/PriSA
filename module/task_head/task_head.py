import torch
import torch.nn as nn
import torch.nn.functional as F
class regression_head(nn.Module):
    def __init__(self,cfg,modality):
        super(regression_head,self).__init__()
        self.cfg = cfg
        if(modality == 'video'):
            self.input_dim = self.cfg.task_v_in_dim
            self.output_dim = self.cfg.task_v_out_dim
            self.hidden_dim = self.cfg.task_v_hid_dim
        elif(modality == 'audio'):
            self.input_dim = self.cfg.task_a_in_dim
            self.output_dim = self.cfg.task_a_out_dim
            self.hidden_dim = self.cfg.task_a_hid_dim
        elif(modality == 'text'):
            self.input_dim = self.cfg.task_t_in_dim
            self.output_dim = self.cfg.task_t_out_dim
            self.hidden_dim = self.cfg.task_t_hid_dim
        else:
            self.input_dim = self.cfg.task_f_in_dim
            self.output_dim = self.cfg.task_f_out_dim
            self.hidden_dim = self.cfg.task_f_hid_dim
        self.linear_1 = nn.Linear(self.input_dim,self.hidden_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.linear_2 = nn.Linear(self.hidden_dim,self.output_dim)
    def forward(self,x):
        x = self.linear_1(x)
        x = self.linear_2(F.relu(self.dropout(x)))
        if(self.output_dim == 1):
            x = x.squeeze(dim=-1)
        return x
class classification_head(nn.Module):
    def __init__(self,cfg,modality):
        super(classification_head,self).__init__()
        self.cfg = cfg
        self.input_dim = self.cfg.MODEL.TASK_HEAD.INPUT_DIM
        self.output_dim = self.cfg.MODEL.TASK_HEAD.OUTPUT_DIM
        self.hidden_dim = self.cfg.MODEL.TASK_HEAD.HIDDEN_DIM
        self.linear_1 = nn.Linear(self.input_dim,self.hidden_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.linear_2 = nn.Linear(self.hidden_dim,self.output_dim)
    def forward(self,x):
        x = self.dropout(x)
        x = self.linear_1(x)
        x = self.linear_2(F.relu(x))
        x = F.softmax(x,dim=-1)
        return x
class multi_classification_head(nn.Module):
    def __init__(self,cfg,modality):
        super(multi_classification_head,self).__init__()
        self.cfg = cfg
        if(modality == 'video'):
            self.input_dim = self.cfg.task_v_in_dim
            self.output_dim = self.cfg.task_v_out_dim
            self.hidden_dim = self.cfg.task_v_hid_dim
        elif(modality == 'audio'):
            self.input_dim = self.cfg.task_a_in_dim
            self.output_dim = self.cfg.task_a_out_dim
            self.hidden_dim = self.cfg.task_a_hid_dim
        elif(modality == 'text'):
            self.input_dim = self.cfg.task_t_in_dim
            self.output_dim = self.cfg.task_t_out_dim
            self.hidden_dim = self.cfg.task_t_hid_dim
        else:
            self.input_dim = self.cfg.task_f_in_dim
            self.output_dim = self.cfg.task_f_out_dim
            self.hidden_dim = self.cfg.task_f_hid_dim
        self.linear_1 = nn.Linear(self.input_dim,self.hidden_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.linear_2 = nn.Linear(self.hidden_dim,self.output_dim)
    def forward(self,x):
        x = self.linear_1(x)
        x = F.relu(self.dropout(x))
        x = self.linear_2(x)
        x = torch.sigmoid(x)
        return x