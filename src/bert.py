import torch.nn as nn
from transformers import BertModel
import torch.nn.functional as F
class BERT(nn.Module):
    def __init__(self,bert_path):
        super(BERT,self).__init__()
        self.model = BertModel.from_pretrained(bert_path)
    def forward(self,input_ids,attention_mask):
        last_hidden_state = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        return last_hidden_state