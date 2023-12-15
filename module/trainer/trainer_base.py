from torch.optim import Adam,SGD
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
from prettytable import PrettyTable
import model
import module.loader as loader
from runx.logx import logx
import math
class trainer_base(object):
    def __init__(self,cfg):
        self.TIMESTAMP = '{0:%Y-%m-%dT%H-%M-%S/}'.format(datetime.now())
        self.cfg = cfg
        self.logger = None
        self.sets = self.cfg.sets
        if(self.cfg.dataset_name == 'mosei_dataset'):
            self.input_list = ['idx(cpu)','vision','audio','text_id','text_mask','text_emb','label','e_label','len']
        elif(self.cfg.dataset_name == 'mosi'):
            self.input_list = ['idx(cpu)','vision','audio','text_id','text_mask','text_emb','label','len']
            self.cfg.v_in_dim = 20
            self.cfg.a_in_dim = 5
        elif(self.cfg.dataset_name == 'sims'):
            self.input_list = ['idx(cpu)','vision','audio','text_id','text_mask','text_emb','label','len']
        elif(self.cfg.dataset_name == 'urfunny'):
            self.input_list = ['idx(cpu)','vision','audio','text_id','text_mask','text_emb','label','len']
        self.task_type = self.cfg.task_head_type
        self.crucial_metric = self.cfg.crucial_metric
        self.decrese_metric = self.cfg.decrese_metric 
        self.best_metric_value = None
        self.best_metric_dict = {}
        self.save_model = self.cfg.save_model
        self.save_model_path = self.cfg.logger_path
        self.best_model_name = None
        self.max_patience = self.cfg.max_patience 
        self.current_patience = self.max_patience 
        self.best_eval_epoch = 0
        self.init_path()
        self.init_criterion()
        self.init_logger()
        if(cfg.init_model):
            self.init_model()
            self.init_optimizer()
            self.init_schedule()
            self.init_metrics()
            self.init_gpu_setting() 
        self.init_dataloader()
        self.report_settings()
    def train(self):
        for epoch in range(self.cfg.epochs):
            train_loss,eval_loss,test_loss = None,None,None
            eval_metric_dict,test_metric_dict = {},{}
            logx.msg('='*30+str(epoch)+'='*30)
            train_loss,train_metric_dict = self.train_one_epoch(epoch)
            self.lr_scheduler.step()
            if('eval' in self.sets):
                eval_loss,eval_metric_dict = self.eval_one_epoch(epoch)
            if('test' in self.sets):
                test_loss,test_metric_dict = self.test_one_epoch(epoch)
            if(self.logger != None):
                self.log_dict['train/train_loss'] = train_loss
                self.log_dict['eval/eval_loss'] = eval_loss
                self.log_dict['test/test_loss'] = test_loss
                for key in eval_metric_dict.keys():
                    self.log_dict[key] = eval_metric_dict[key]
                for key in test_metric_dict.keys():
                    self.log_dict[key] = test_metric_dict[key]
                self.write_logger(epoch)
            if(self.current_patience<0):
                break
        self.end_logger()
    def batch2dict(self,batch):
        input_dict = {}
        n = len(batch)
        for i,name in enumerate(self.input_list):
            input_dict[name] = batch[i]
        return input_dict
    def train_one_epoch(self,epoch):
        pass
    def eval_one_epoch(self,epoch):
        pass
    def test_one_epoch(self,epoch):
        pass
    def init_gpu_setting(self):
        self.use_cuda = torch.cuda.is_available()
        if self.cfg.gpu_ids == "auto":
            gpu_num = torch.cuda.device_count()
            self.gpu_ids = list(range(gpu_num))
            print(f'gpu set auto, finding {gpu_num} gpus, let\'s use them')
        elif self.cfg.gpu_ids != "auto" and self.cfg.gpu_ids != "":
            self.gpu_ids = [int(gpu_id) for gpu_id in self.cfg.gpu_ids.split(",")]
            print("gpu is set to %s, let's use them", self.gpu_ids)
        else:
            self.use_cuda = False
        if(self.use_cuda):
            self.model = self.model.cuda()
            self.model = torch.nn.DataParallel(self.model,device_ids=self.gpu_ids)
    def init_criterion(self):
        self.criterion_list = {'MSELoss':nn.MSELoss(),'CrossEntropyLoss':nn.CrossEntropyLoss(),'BCELoss':nn.BCELoss()}
        self.criterion = self.criterion_list[self.cfg.criterion]
    def init_optimizer(self):
        self.optimizer_dict = {'adam':Adam,'sgd':SGD}
        self.optimizer = self.optimizer_dict[self.cfg.optimizer](params=self.model.parameters(),lr=self.cfg.lr)
    def init_schedule(self):
        warm_up_epochs = 10
        max_num_epochs = self.cfg.epochs
        lr_milestones = [20,80]
        warm_up_with_multistep_lr = lambda epoch: (epoch+1) / warm_up_epochs if epoch < warm_up_epochs else 0.1**len([m for m in lr_milestones if m <= epoch])
        warm_up_with_cosine_lr = lambda epoch: (epoch+1) / warm_up_epochs if epoch < warm_up_epochs \
        else 0.5 * ( math.cos((epoch - warm_up_epochs) /(max_num_epochs - warm_up_epochs) * math.pi) + 1)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,lr_lambda=warm_up_with_multistep_lr)
    def init_path(self):
        if(self.cfg.logger):
            self.logger_path = self.cfg.logger_path
            os.makedirs(self.logger_path,exist_ok=True)
        if(self.cfg.save_model):
            self.save_model_path = self.cfg.logger_path
            os.makedirs(self.save_model_path,exist_ok=True)
        self.load_model_path = self.cfg.load_model_path
    def init_logger(self):
        if(self.cfg.logger):
            self.logger = SummaryWriter(self.logger_path)
        self.log_dict = {}
    def report_step(self):
        pass
    def report_settings(self):
        print('logger save in ',self.cfg.logger_path)
        self.table = PrettyTable()
        self.table.field_names = ['parameter name','value']
        self.table.add_row(['gpu_nums',len(self.gpu_ids)])
        self.table.add_row(['batch size',self.batch_size])
        self.table.add_row(['dataset',self.dataset_name])
        self.table.add_row(['model',self.model_name])
        self.table.add_row(['task',self.task_type])
        self.table.add_row(['optimizer',self.cfg.optimizer])
        self.table.add_row(['criterion',self.cfg.criterion])
        print(self.table)
    def init_dataloader(self):
        self.batch_size = batch_size = self.cfg.batch_size * len(self.gpu_ids)
        self.dataset_name = dataset_name = self.cfg.dataset_name
        num_workers = self.cfg.num_workers
        pin_memory = self.cfg.pin_memory
        self.train_dataset = train_dataset = loader.get_instance(dataset_name,self.cfg,mode='train')
        self.valid_dataset = valid_dataset = loader.get_instance(dataset_name,self.cfg,mode='valid')
        self.test_dataset = test_dataset=  loader.get_instance(dataset_name,self.cfg,mode='test')
        self.train_loader = DataLoader(train_dataset,
                                       batch_size=batch_size,
                                       shuffle = True,
                                       num_workers = num_workers,
                                       pin_memory = pin_memory,
                                       collate_fn=train_dataset.collate_fn,
                                       drop_last=self.cfg.drop_last_batch)
        self.valid_loader = DataLoader(valid_dataset,batch_size=batch_size,collate_fn=valid_dataset.collate_fn)
        self.test_loader = DataLoader(test_dataset,batch_size=batch_size,collate_fn=test_dataset.collate_fn)
    def init_model(self):
        self.model_name =  self.cfg.model_name
        self.model = model.get_instance(self.model_name,self.cfg)
        if(self.load_model_path != ''):
            self.model.load_state_dict(torch.load(self.load_model_path))
    def init_metrics(self):
        self.metrics = None
    def auto_to_gpu(self,input_dict):
        output_dict = {}
        if(self.use_cuda):
            for key in input_dict.keys():
                output_dict[key] = input_dict[key].cuda()
            return output_dict
        else:
            return input_dict
    def write_logger(self,epoch):
        for key in self.log_dict.keys():
            phase = key.split('/')[0]
            modality = key.split('/')[1]
            metric = key.split('/')[-1]
            value = self.log_dict[key]
            self.logger.add_scalar(key,value,epoch)
            if(phase == 'eval' and modality=='fusion' and metric == self.crucial_metric):
                if(epoch==0):
                    self.best_metric_value = value
                else:
                    if(self.decrese_metric):
                        signal =  self.best_metric_value>value
                        if(signal):
                            self.current_patience = self.max_patience
                            self.best_eval_epoch = epoch
                        else:
                            self.current_patience -= 1
                        self.best_metric_value = value if signal else self.best_metric_value
                        if(signal and self.save_model):
                            self.best_eval_epoch = epoch
                            self.save_model_state(epoch,value)
                    else:
                        signal =  self.best_metric_value<value
                        if(signal):
                            self.current_patience = self.max_patience
                            self.best_eval_epoch = epoch
                        else:
                            self.current_patience -= 1
                        self.best_metric_value = value if signal else self.best_metric_value
                        if(signal and self.save_model):
                            self.best_eval_epoch = epoch
                            self.save_model_state(epoch,value)
            if(phase == 'test'and modality=='fusion' and epoch == self.best_eval_epoch):
                self.best_metric_dict[metric] = value
    def end_logger(self):
        if(self.logger is not None):
            print(f'[{self.best_eval_epoch}] best results: ')
            table = PrettyTable()
            metric_names = list(self.best_metric_dict.keys())
            table.field_names = ['metric_name','values']
            for i,k in enumerate(metric_names):
                table.add_row([k,f'{self.best_metric_dict[k]:.3}'])
            logx.msg(table.get_string())
            self.logger.close()
    def save_model_state(self,epoch,value):
        self.model.eval()
        if(self.best_model_name is not None):
            os.remove(os.path.join(self.save_model_path,self.best_model_name)) 
        self.best_model_name = f'{epoch}_{value:.3f}.pt'
        save_model_path_epoch = os.path.join(self.save_model_path,self.best_model_name)
        torch.save(self.model.module.state_dict(),save_model_path_epoch)