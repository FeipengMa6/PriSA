import torch
import pickle
from torch.utils.data.dataset import Dataset
import numpy as np
class basic_dataset(Dataset):
    def __init__(self,cfg,mode) -> None:
        self.data_path = cfg.data_path
        with open(self.data_path,'rb') as f:
            self.data = pickle.load(f)[mode]
        self.len = len(self.data['vision'])
        self.e_label2id = {'anger':0,'disgust':1,'fear':2,'happy':3,
                        'sad':4,'surprise':5}
    def __getitem__(self,idx):
        vision = torch.from_numpy(self.data['vision'][idx]).float()
        audio = torch.from_numpy(self.data['audio'][idx]).float()
        text_emb = torch.from_numpy(self.data['text'][idx]).float()
        text_id = torch.from_numpy(self.data['text_bert'][idx][0])
        text_mask = torch.from_numpy(self.data['text_bert'][idx][1])
        r_label = np.float32(self.data['regression_labels'][idx])
        e_label = self.data['emotion_labels'][idx]
        e_label[e_label>0] = 1
        e_label = torch.from_numpy(self.data['emotion_labels'][idx])
        v_len = torch.sum(vision,dim=-1)
        v_len = len(v_len[v_len!=0]) 
        a_len = torch.sum(audio,dim=-1)
        a_len = len(a_len[a_len!=0]) 
        t_len = self.data['text_bert'][idx][1]
        t_len = len(t_len[t_len!=0]) 
        vat_len = [v_len,a_len,t_len]
        return idx,vision,audio,text_id,text_mask,text_emb,r_label,e_label,vat_len
    def __len__(self):
        return self.len
    @staticmethod
    def _truncate(v_stack,a_stack,vat_stack):
        v_max_len = max(vat_stack[:,0])
        a_max_len = max(vat_stack[:,1])
        v_stack = v_stack[:,:v_max_len,:]
        a_stack = a_stack[:,:a_max_len,:]
        return v_stack,a_stack
    @staticmethod
    def collate_fn(batch):
        idx,vision,audio,text_id,text_mask,text_emb,r_label,e_label,vat_len = zip(*batch)
        vat_l_stack = np.array(vat_len)
        vat_l_stack = torch.from_numpy(vat_l_stack)
        v_stack = torch.stack(vision,dim=0)
        a_stack = torch.stack(audio,dim=0)
        ti_stack = torch.stack(text_id,dim=0)
        tm_stack = torch.stack(text_mask,dim=0)
        te_stack = torch.stack(text_emb,dim=0)
        rl_stack = torch.tensor(r_label)
        el_stack = torch.stack(e_label)
        return idx,v_stack,a_stack,ti_stack,tm_stack,te_stack,rl_stack,el_stack,vat_l_stack
class mosi_dataset(Dataset):
    def __init__(self,cfg,mode) -> None:
        self.data_path = cfg.data_path
        with open(self.data_path,'rb') as f:
            self.data = pickle.load(f)[mode]
        self.len = len(self.data['vision'])
        self.e_label2id = {'anger':0,'disgust':1,'fear':2,'happy':3,
                        'sad':4,'surprise':5}
    def __getitem__(self,idx):
        vision = torch.from_numpy(self.data['vision'][idx]).float()
        audio = torch.from_numpy(self.data['audio'][idx]).float()
        text_emb = torch.from_numpy(self.data['text'][idx]).float()
        text_id = torch.from_numpy(self.data['text_bert'][idx][0])
        text_mask = torch.from_numpy(self.data['text_bert'][idx][1])
        r_label = np.float32(self.data['regression_labels'][idx])
        v_len = torch.sum(vision,dim=-1)
        v_len = len(v_len[v_len!=0]) 
        a_len = torch.sum(audio,dim=-1)
        a_len = len(a_len[a_len!=0]) 
        t_len = self.data['text_bert'][idx][1]
        t_len = len(t_len[t_len!=0]) 
        vat_len = [v_len,a_len,t_len]
        return idx,vision,audio,text_id,text_mask,text_emb,r_label,vat_len
    def __len__(self):
        return self.len
    @staticmethod
    def collate_fn(batch):
        idx,vision,audio,text_id,text_mask,text_emb,r_label,vat_len = zip(*batch)
        v_stack = torch.stack(vision,dim=0)
        a_stack = torch.stack(audio,dim=0)
        ti_stack = torch.stack(text_id,dim=0)
        tm_stack = torch.stack(text_mask,dim=0)
        te_stack = torch.stack(text_emb,dim=0)
        rl_stack = torch.tensor(r_label)
        vat_l_stack = np.array(vat_len)
        vat_l_stack = torch.from_numpy(vat_l_stack)
        return idx,v_stack,a_stack,ti_stack,tm_stack,te_stack,rl_stack,vat_l_stack

class urfunny_dataset(Dataset):
    def __init__(self,cfg,mode) -> None:
        self.data_path = cfg.data_path
        with open(self.data_path,'rb') as f:
            self.data = pickle.load(f)[mode]
        self.len = len(self.data['vision'])
    def __getitem__(self,idx):
        vision = torch.from_numpy(self.data['vision'][idx]).float()
        audio = torch.from_numpy(self.data['audio'][idx]).float()
        text_emb = torch.from_numpy(self.data['text'][idx]).float()
        text_id = torch.from_numpy(self.data['text_bert'][idx][0])
        text_mask = torch.from_numpy(self.data['text_bert'][idx][1])
        label = self.data['classification_labels'][idx]
        v_len = torch.sum(vision,dim=-1)
        v_len = len(v_len[v_len!=0]) 
        a_len = torch.sum(audio,dim=-1)
        a_len = len(a_len[a_len!=0]) 
        t_len = self.data['text_bert'][idx][1]
        t_len = len(t_len[t_len!=0]) 
        vat_len = [v_len,a_len,t_len]
        return idx,vision,audio,text_id,text_mask,text_emb,label,vat_len
    def __len__(self):
        return self.len
    @staticmethod
    def collate_fn(batch):
        idx,vision,audio,text_id,text_mask,text_emb,label,vat_len = zip(*batch)
        v_stack = torch.stack(vision,dim=0)
        a_stack = torch.stack(audio,dim=0)
        ti_stack = torch.stack(text_id,dim=0)
        tm_stack = torch.stack(text_mask,dim=0)
        te_stack = torch.stack(text_emb,dim=0)
        l_stack = torch.tensor(label).to(torch.int64)
        vat_l_stack = np.array(vat_len)
        vat_l_stack = torch.from_numpy(vat_l_stack)
        return idx,v_stack,a_stack,ti_stack,tm_stack,te_stack,l_stack,vat_l_stack