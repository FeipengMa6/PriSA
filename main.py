import module.trainer as Trainer
import argparse
from runx.logx import logx
from coolname import generate_slug
import os
import torch
import numpy as np
import random
parser = argparse.ArgumentParser(description='MOSEI Sentiment Analysis')
parser.add_argument('--dataset_name', default='mosei_dataset', type=str,
                    help='name of the dataset')
parser.add_argument('--num_workers', default=2, type=int,
                    help='num workers of DataLoader')    
parser.add_argument('--pin_memory', action='store_false',
                    help='if call this parameter, pin_memory will turn to false, default is true')
parser.add_argument('--batch_size','--B', default=64, type=int,
                    help='batch_size  of DataLoader')    
parser.add_argument('--data_path', default='./data/mosei_unaligned_50.pkl', type=str,
                    help='path of mosei dataset files')   
parser.add_argument('--unaligned',action='store_true',default=True, help='default using align feature,if call this argument, trainer will use unaligned mosei data')
parser.add_argument('--modality', nargs='+', default=['audio','video','fusion'],
                    help='the modalities we will train in the trainer')  
parser.add_argument('--trainer_name', default='prisa_trainer',type=str) 
parser.add_argument('--model_name', default='prisa_model',type=str) 
parser.add_argument('--sets', default=['train','eval','test'],type=list)
parser.add_argument('--logger_path', default='./logs',type=str)        
parser.add_argument('--save_model_path', default='./logs',type=str)
parser.add_argument('--save_model', action='store_true', default=True,
                    help='if call the parameter, trainer will save model, default not save model')
parser.add_argument('--logger', action='store_true', default=True)                     
parser.add_argument('--gpu_ids', default='0',type=str)        
parser.add_argument('--init_model', action='store_false')            
parser.add_argument('--optimizer', default='adam',type=str) 
parser.add_argument('--text_lr', default=1e-5,type=float,help='lr of bert model')
parser.add_argument('--audio_lr', default=1e-3,type=float,help='lr of audio classify head') 
parser.add_argument('--video_lr', default=1e-4,type=float,help='lr of video classify head') 
parser.add_argument('--fusion_head_lr', default=1e-5,type=float,help='lr of fusion regression head') 
parser.add_argument('--base_lr', default=1e-4,type=float,help='lr of video and audio backbone') 
parser.add_argument('--task_head_type', default='regression',type=str,help='regression or classification') 
parser.add_argument('--criterion', default='L1Loss',type=str,help='if task is classification, should use BCELoss')  
parser.add_argument('--epochs', default=100,type=int)  
parser.add_argument('--bert_path', default='./pretrained_models/bert-base-uncased',type=str) 
parser.add_argument('--embed_dim', default=128,type=int) 
parser.add_argument('--v_in_dim', default=35,type=int)  
parser.add_argument('--v_hid_dim', default=128,type=int)  
parser.add_argument('--v_out_dim', default=128,type=int)  
parser.add_argument('--a_in_dim', default=74,type=int)  
parser.add_argument('--a_hid_dim', default=128,type=int)  
parser.add_argument('--a_out_dim', default=128,type=int)  
parser.add_argument('--t_in_dim', default=768,type=int)  
parser.add_argument('--t_hid_dim', default=128,type=int)  
parser.add_argument('--t_out_dim', default=128,type=int) 
parser.add_argument('--layer', default=1,type=int) 
parser.add_argument('--num_layers_2', default=3,type=int)
parser.add_argument('--private_layers', default=3,type=int)
parser.add_argument('--num_heads', default=4,type=int) 
parser.add_argument('--dout_p', default=0.1,type=float) 
parser.add_argument('--task_v_in_dim', default=128,type=int)  
parser.add_argument('--task_v_hid_dim', default=128,type=int)  
parser.add_argument('--task_v_out_dim', default=1,type=int)  
parser.add_argument('--task_a_in_dim', default=128,type=int)  
parser.add_argument('--task_a_hid_dim', default=128,type=int)  
parser.add_argument('--task_a_out_dim', default=1,type=int)  
parser.add_argument('--task_t_in_dim', default=128,type=int)  
parser.add_argument('--task_t_hid_dim', default=1024,type=int)  
parser.add_argument('--task_t_out_dim', default=1,type=int) 
parser.add_argument('--task_f_in_dim', default=512,type=int)  
parser.add_argument('--task_f_hid_dim', default=512,type=int)  
parser.add_argument('--task_f_out_dim', default=1,type=int) 
parser.add_argument('--k', default=64,type=int)
parser.add_argument('--seed', default=2021,type=int)
parser.add_argument('--using_bert', action='store_true', default=True)
parser.add_argument('--crucial_metric', default='mae',type=str)
parser.add_argument('--decrese_metric', action='store_true', default=True)
parser.add_argument('--report_metric',default='non0_f1-score')
parser.add_argument('--load_model_path',default='',type=str)
parser.add_argument('--attn_dropout', type=float, default=0.1)
parser.add_argument('--attn_dropout_a', type=float, default=0.0)
parser.add_argument('--attn_dropout_v', type=float, default=0.0)
parser.add_argument('--relu_dropout', type=float, default=0.1)
parser.add_argument('--embed_dropout', type=float, default=0.25)
parser.add_argument('--res_dropout', type=float, default=0.1)
parser.add_argument('--num_layers_sa', type=int, default=3)
parser.add_argument('--rnn_type', type=str, default='LSTM')
parser.add_argument('--num_layers_ss', type=int, default='2')
parser.add_argument('--num_directions_ss', type=int, default=1)
parser.add_argument('--input_size_ss', type=int, default=128*2)
parser.add_argument('--hidden_size_ss', type=int, default=128*2)
parser.add_argument('--rnn_dropout', type=float, default=0.1)
parser.add_argument('--attn_mask', action='store_true', default=True)   
parser.add_argument('--T', type=float, default=0.1)
parser.add_argument('--percent', type=float, default=0.5)
parser.add_argument('--sample_p', type=float, default=0.1)
parser.add_argument('--pretrain', action='store_true')
parser.add_argument('--pretrain_epochs', type=int,default=81)
parser.add_argument('--alpha', type=float,default=0.01)
parser.add_argument('--max_patience', type=int,default=3)
parser.add_argument('--beta', type=float,default=0.01)
parser.add_argument('--max_norm',type=float,default=20)
parser.add_argument('--contrastive_learning',action='store_true', default=True)
parser.add_argument('--drop_last_batch',action='store_true', default=True)
parser.add_argument('--process_not_visible',action='store_true')
parser.add_argument('--aug',type=str,default='reverse_multiscale')
parser.add_argument('--crop_scale',type=float,default=0.08)
parser.add_argument('--cutout_scale',type=float,default=0.1)
parser.add_argument('--reverse_p',type=float,default=0.1)
parser.add_argument('--stride',type=int,default=3)
parser.add_argument('--multiscale_p',type=float,default=0.08)
parser.add_argument('--disorder_p',type=float,default=0.1)
parser.add_argument('--d',type=float,default=0.9)
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
if __name__ == '__main__':
    args = parser.parse_args()
    logger_folder_name = generate_slug(2)
    setup_seed(args.seed)
    args.logger_path = os.path.join(args.logger_path)
    logx.initialize(logdir=args.logger_path, coolname=True, tensorboard=True,hparams=vars(args))
    trainer = Trainer.get_instance(args.trainer_name,args)
    print(args)
    trainer.train()    
    print(logger_folder_name)