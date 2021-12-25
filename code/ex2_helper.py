# !pip install transformers
# !pip install sentencepiece
bert_dir = '.'
data_dir = '../milestone3/Liar-Plus'
output_dir = '.'
from transformers import XLNetTokenizer, XLNetModel, AdamW, get_cosine_schedule_with_warmup

import warnings
warnings.filterwarnings("ignore")
from torch.utils.data import TensorDataset,DataLoader
# !pip install datasets
import pandas as pd
import argparse
import numpy as np
from collections import Counter
# from datasets import load_dataset
import os
import torch
import pickle
import re
import time
import copy
from torch.utils.data import DataLoader, Dataset
import torch.optim as optimizer 
from torch import nn
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import copy
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = "sans-serif"
plt.rcParams['font.sans-serif'] = ['Times New Roman']
sns.set_style("whitegrid")
sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})
sns.set(font_scale=1.2)
import argparse

class XLNet_Model(nn.Module):
    def __init__(self, num_classes=6, alpha=0.5):
        self.alpha = alpha
        super(XLNet_Model, self).__init__()
        self.net = XLNetModel.from_pretrained(bert_dir)
        
        # for name,param in self.net.named_parameters():
        #     param.requires_grad=True
        ## keep some of layers fixed when training
        for name, param in self.net.named_parameters():
            if 'layer.11' in name or 'layer.10' in name or 'layer.9' in name or 'layer.8' in name or 'pooler.dense' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        self.fc = nn.Linear(768,6)

    def forward(self, x):
        x = x.long()
        x = self.net(x, output_all_encoded_layers=False).last_hidden_state
        x = F.dropout(x, self.alpha, training=self.training)
        x = torch.max(x, dim=1)[0]
        
        x = self.fc(x)
        # return torch.sigmoid(x)
        return x
    


def clean_text(w):
  if type(w)==float:
    return " "
  return re.sub(
          r"([.,'!?\"()*#:;])",
          '',
          w.lower()
          ).replace('/', ' ')
def preprocessing(train=False, eval=False, test=False):
  cols = ['id','label','statement','subject','speaker','job_title','state_info',
          'party_affiliation','barely_true_counts','false_counts',
          'half_true_counts','mostly_true_counts','pants_on_fire_counts',
          'context','justification']
  label_dict = {"false" : 0, "half-true" : 1, "mostly-true" : 2, "true": 3, "barely-true" : 4, "pants-fire" : 5 } 
  def get_label(x):
    if x not in label_dict:
      return 1
    return label_dict[x]
  if train:
    dst_path = os.path.join(data_dir,'train2.tsv')
  if eval:
    dst_path = os.path.join(data_dir,'val2.tsv')
  if test:
    dst_path = os.path.join(data_dir,'test2.tsv')
  current_dataset = pd.read_csv(dst_path, sep='\t', header = None, names=cols)
  current_dataset['label'] = current_dataset['label'].apply(lambda x: get_label(x))
  current_dataset.reset_index(drop=True,inplace=True)
  return current_dataset

meta_cols = ['subject','speaker','job_title','state_info','party_affiliation','context','justification']
def get_meta_embed(dst, meta_cols):
  all_text = []
  for i in range(len(dst)):
    cur = ''
    for c in meta_cols:
      try:
        cur += str(dst[c][i]) + ' SS '
      except:
        print(c,i)
        return
    all_text.append(cur)
  return all_text

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
def evaluate_p_r_f1_acc(y_pred, y_true):
    precision = precision_score(y_pred, y_true)
    recall = recall_score(y_pred, y_true)
    fscore = f1_score(y_pred, y_true)
    acc = accuracy_score(y_pred, y_true)
    return precision, recall, fscore, acc
def category_from_output(output):
  res = []
  for i in output:
    top_n, top_i = i.topk(1)
    category_i = top_i[0].item()
    res.append(category_i)
  return res



