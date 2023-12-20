from module.trainer.trainer_base import trainer_base
from torch.optim import Adam,SGD
from module.metrics.mosei_metrics import mosei_metrics,sims_metrics,urfunny_metrics
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
from runx.logx import logx
import math
from module.loss.loss_base import NCELoss
import sys
class prisa_trainer(trainer_base):
    def __init__(self,cfg):
        super(prisa_trainer,self).__init__(cfg)
        self.target_class_idx = 0
        self.modality_weight = {'video':1,'audio':1,'text':1,'fusion':1}
    def init_metrics(self):
        if(self.cfg.dataset_name == 'sims'):
            self.metrics = sims_metrics
        elif(self.cfg.dataset_name == 'urfunny'):
            self.metrics = urfunny_metrics
        else:
            self.metrics = mosei_metrics
    def init_optimizer(self):
        self.optimizer_dict = {'adam':Adam,'sgd':SGD}
        if(self.cfg.using_bert):
            bert_params = list(map(id,self.model.t_model.parameters()))
            vt_params = list(map(id,self.model.vt_layers.parameters()))
            at_params = list(map(id,self.model.at_layers.parameters()))
            base_params = filter(lambda p: id(p) not in (bert_params+vt_params+at_params),self.model.parameters())
            params_group = [{'params': base_params},
                            {'params': self.model.t_model.parameters(),'lr': self.cfg.text_lr},
                            {'params': self.model.vt_layers.parameters(),'lr': self.cfg.video_lr},
                            {'params': self.model.at_layers.parameters(),'lr': self.cfg.audio_lr}]
            self.optimizer = self.optimizer_dict[self.cfg.optimizer](params=params_group,lr=self.cfg.base_lr)
        else:
            self.optimizer = self.optimizer_dict[self.cfg.optimizer](params=self.model.parameters(),lr=self.cfg.base_lr)
    def init_schedule(self):
        warm_up_epochs = 5
        max_num_epochs = self.cfg.epochs
        lr_milestones = [20,50]
        warm_up_with_multistep_lr = lambda epoch: (epoch+1) / warm_up_epochs if epoch < warm_up_epochs else 0.1**len([m for m in lr_milestones if m <= epoch])
        warm_up_with_cosine_lr = lambda epoch: (epoch+1) / warm_up_epochs if epoch < warm_up_epochs \
        else 0.5 * ( math.cos((epoch - warm_up_epochs) /(max_num_epochs - warm_up_epochs) * math.pi) + 1)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,lr_lambda=warm_up_with_multistep_lr)
    def auto_to_gpu(self,input_dict):
        output_dict = {}
        if(self.use_cuda):
            for key in input_dict.keys():
                if('cpu' not in key):
                    output_dict[key] = input_dict[key].cuda()
                else:
                    output_dict[key] = input_dict[key]
            return output_dict
        else:
            return input_dict
    def init_criterion(self):
        self.criterion_list = {'MSELoss':nn.MSELoss(),'CrossEntropyLoss':nn.CrossEntropyLoss(),'BCELoss':nn.BCELoss(),'L1Loss':nn.L1Loss(),'SmoothL1Loss':nn.SmoothL1Loss()}
        self.criterion = self.criterion_list[self.cfg.criterion]
        self.cl_criterion = NCELoss(temperature=self.cfg.T,d=self.cfg.d)
    def label_conversion(self,label):
        target = label>0
        target = target.long()
        return target
    def directly_save_model(self,epoch):
        self.model.eval()
        if(self.best_model_name is not None):
            os.remove(os.path.join(self.save_model_path,self.best_model_name)) 
        self.best_model_name = f'best_model.pt'
        self.save_model_path_epoch = os.path.join(self.save_model_path,self.best_model_name)
        torch.save(self.model.module.state_dict(),self.save_model_path_epoch)
    def pretrain(self):
        for epoch in range(self.cfg.pretrain_epochs):
            logx.msg('='*30+str(epoch)+'='*30)
            train_loss = self.pretrain_one_epoch(epoch)
            self.lr_scheduler.step()
        self.init_optimizer()
        self.init_schedule()
    def pretrain_one_epoch(self,epoch):
        phase = 'train'
        self.model.train()
        input_dict = {}
        y_true, y_pred = {}, {}
        losses = []
        c_loss =[]
        msg_string = ''
        for batch in tqdm(self.train_loader):
            self.model.zero_grad()
            input_dict = self.batch2dict(batch)
            input_dict = self.auto_to_gpu(input_dict)
            score_dict,state_dict = self.model(input_dict)
            loss  = state_dict['loss']
            loss.backward()
            self.optimizer.step()
            c_loss.append(state_dict['loss'].item())
        c_loss_mean = np.mean(c_loss)
        msg_string = f'c_loss={c_loss_mean:.3f} '
        logx.msg(msg_string)
        return c_loss_mean
    def train_one_epoch(self,epoch):
        phase = 'train'
        self.model.train()
        input_dict = {}
        y_true, y_pred = {}, {}
        losses = []
        a_at_losses =[]
        v_vt_losses = []
        at_vt_losses = []
        msg_string = ''
        for batch in self.train_loader:
            self.model.zero_grad()
            input_dict = self.batch2dict(batch)
            input_dict = self.auto_to_gpu(input_dict)
            score_dict,state_dict = self.model(input_dict)
            loss = None
            for key in self.cfg.modality:
                if(loss is None):
                    loss = self.modality_weight[key]*self.criterion(score_dict[key],input_dict['label'])
                else:
                    loss += self.modality_weight[key]*self.criterion(score_dict[key],input_dict['label'])
            task_loss = loss.item()
            v_private_loss = self.criterion(score_dict['video_private'],input_dict['label'])
            a_private_loss = self.criterion(score_dict['audio_private'],input_dict['label'])
            loss += v_private_loss+a_private_loss
            target = input_dict['label'] 
            v_vt_loss = self.cfg.alpha * (self.cl_criterion(state_dict['v_private'],state_dict['vt_fusion'],target) + self.cl_criterion(state_dict['vt_fusion'],state_dict['v_private'],target))
            a_at_loss = self.cfg.alpha * (self.cl_criterion(state_dict['a_private'],state_dict['at_fusion'],target) + self.cl_criterion(state_dict['at_fusion'],state_dict['a_private'],target))
            at_vt_loss = self.cfg.beta * (self.cl_criterion(state_dict['vt_fusion'],state_dict['at_fusion'],target) + self.cl_criterion(state_dict['at_fusion'],state_dict['vt_fusion'],target))
            if(self.cfg.contrastive_learning):
                loss += at_vt_loss
            for key in self.cfg.modality:
                if(key not in y_pred.keys()):
                    y_pred[key] = []
                    y_true[key] = []
                if(self.cfg.dataset_name == 'urfunny'):
                    y_pred[key].append(score_dict[key].argmax(dim=1).detach().cpu().numpy())
                else:
                    y_pred[key].append(score_dict[key].detach().cpu().numpy())
                y_true[key].append(input_dict['label'].detach().cpu().numpy())
            loss.backward()
            self.optimizer.step()
            losses.append(task_loss)
            a_at_losses.append(a_at_loss.item())
            v_vt_losses.append(v_vt_loss.item())
            at_vt_losses.append(at_vt_loss.item())
        loss_mean = np.mean(losses) 
        a_at_mean = np.mean(a_at_losses) 
        v_vt_mean = np.mean(v_vt_losses)
        at_vt_mean = np.mean(at_vt_losses)
        for key in y_true.keys():
            y_true[key] = np.concatenate(y_true[key], axis=0).squeeze()
            y_pred[key] = np.concatenate(y_pred[key], axis=0).squeeze()
        if(self.cfg.dataset_name == 'urfunny'):
            target_names = None
        else:
            target_names = list(self.valid_dataset.e_label2id.keys())
        metric_dict = {} 
        for key in y_true.keys():
            metric_dict.update(self.metrics(y_true[key],y_pred[key],target_names=target_names,key_head=phase+'/'+key)) 
        msg_string = f'train loss={loss_mean:.3f} a_at_loss={a_at_mean} v_vt_loss={v_vt_mean} at_vt_loss={at_vt_mean} '
        for key in y_pred.keys():
            m = metric_dict[phase+'/'+key+'/'+'mae']
            msg_string += f'{key}:{m:.3f} '
            m = metric_dict[phase+'/'+key+'/'+self.cfg.report_metric]
            msg_string += f'{m:.3f} '
        logx.msg(msg_string)
        return loss_mean,metric_dict
    def eval_one_epoch(self,epoch):
        phase = 'eval'
        self.model.eval()
        input_dict = {}
        y_true, y_pred = {}, {}
        losses = []
        msg_string = ''
        with torch.no_grad():
            for batch in self.valid_loader:
                input_dict = self.batch2dict(batch)
                input_dict = self.auto_to_gpu(input_dict)
                score_dict,state_dict = self.model(input_dict)
                loss = None
                for key in self.cfg.modality:
                    if(loss is None):
                        loss = self.modality_weight[key]*self.criterion(score_dict[key],input_dict['label'])
                    else:
                        loss += self.modality_weight[key]*self.criterion(score_dict[key],input_dict['label'])
                losses.append(loss.item())
                for key in self.cfg.modality:
                    if(key not in y_pred.keys()):
                        y_pred[key] = []
                        y_true[key] = []
                    if(self.cfg.dataset_name == 'urfunny'):
                        y_pred[key].append(score_dict[key].argmax(dim=1).detach().cpu().numpy())
                    else:
                        y_pred[key].append(score_dict[key].detach().cpu().numpy())
                    y_true[key].append(input_dict['label'].detach().cpu().numpy())
        eval_loss = np.mean(losses)
        for key in y_true.keys():
            y_true[key] = np.concatenate(y_true[key], axis=0).squeeze()
            y_pred[key] = np.concatenate(y_pred[key], axis=0).squeeze()
        if(self.cfg.dataset_name == 'urfunny'):
            target_names = None
        else:
            target_names = list(self.valid_dataset.e_label2id.keys())
        metric_dict = {}
        for key in y_true.keys():
            metric_dict.update(self.metrics(y_true[key],y_pred[key],target_names=target_names,key_head=phase+'/'+key)) 
        msg_string = f'eval_loss={eval_loss:.3f} '
        for key in y_pred.keys():
            m = metric_dict[phase+'/'+key+'/'+'mae']
            msg_string += f'{key}:{m:.3f} '
            m = metric_dict[phase+'/'+key+'/'+self.cfg.report_metric]
            msg_string += f'{m:.3f} '
        logx.msg(msg_string)
        return eval_loss,metric_dict
    def test_one_epoch(self,epoch):
        phase = 'test'
        self.model.eval()
        input_dict = {}
        y_true, y_pred = {}, {}
        losses = []
        msg_string = ''
        with torch.no_grad():
            for batch in self.test_loader:
                input_dict = self.batch2dict(batch)
                input_dict = self.auto_to_gpu(input_dict)
                score_dict,state_dict = self.model(input_dict)
                loss = None
                for key in self.cfg.modality:
                    if(loss is None):
                        loss = self.modality_weight[key]*self.criterion(score_dict[key],input_dict['label'])
                    else:
                        loss += self.modality_weight[key]*self.criterion(score_dict[key],input_dict['label'])
                losses.append(loss.item())
                for key in self.cfg.modality:
                    if(key not in y_pred.keys()):
                        y_pred[key] = []
                        y_true[key] = []
                    if(self.cfg.dataset_name == 'urfunny'):
                        y_pred[key].append(score_dict[key].argmax(dim=1).detach().cpu().numpy())
                    else:
                        y_pred[key].append(score_dict[key].detach().cpu().numpy())
                    y_true[key].append(input_dict['label'].detach().cpu().numpy())
        test_loss = np.mean(losses)
        for key in y_true.keys():
            y_true[key] = np.concatenate(y_true[key], axis=0).squeeze()
            y_pred[key] = np.concatenate(y_pred[key], axis=0).squeeze()
        if(self.cfg.dataset_name == 'urfunny'):
            target_names = None
        else:
            target_names = list(self.valid_dataset.e_label2id.keys())
        metric_dict = {}
        if(epoch-1==self.best_eval_epoch):
            logx.metric(phase='val',metrics=self.logx_metric_dict,epoch=self.best_eval_epoch)
        self.logx_metric_dict = {}
        for key in y_true.keys():
            metric = self.metrics(y_true[key],y_pred[key],target_names=target_names,key_head=phase+'/'+key)
            metric_dict.update(metric)
            if('fusion' in key):
                for k in metric.keys():
                    self.logx_metric_dict[k.split('/')[-1]] = metric[k]
        msg_string = f'test_loss={test_loss:.3f} '
        for key in y_pred.keys():
            m = metric_dict[phase+'/'+key+'/'+'mae']
            msg_string += f'{key}:{m:.3f} '
            m = metric_dict[phase+'/'+key+'/'+self.cfg.report_metric]
            msg_string += f'{m:.3f} '
        logx.msg(msg_string)
        return test_loss,metric_dict