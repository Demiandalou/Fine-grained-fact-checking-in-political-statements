from torch.utils.data import DataLoader, Dataset

import torch.optim as optimizer 
import torch.nn.functional as F
from torch import nn

import gensim
import pandas as pd
import argparse
import numpy as np
from collections import Counter
from datasets import load_dataset
import os
import torch
import pickle
import re
import time
import copy
import math

from sklearn.metrics import accuracy_score, f1_score
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = "sans-serif"
plt.rcParams['font.sans-serif'] = ['Times New Roman']
sns.set_style("whitegrid")
sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})
sns.set(font_scale=1.2)
data_dir = '.'

# 2368-2182
class text_only_CNN(nn.Module):
    def __init__(self, token_size, pretrained_emb):
        super(text_only_CNN, self).__init__()
        num_class = 6
        dropout_rate = 0.5
        self.ksizes = [5,5,5]
        print(dropout_rate,self.ksizes)
        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=300
        )
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))
        self.conv_unit = nn.Sequential(
            nn.Conv1d(in_channels=300, out_channels=128, kernel_size=self.ksizes[0]),
            nn.Dropout(dropout_rate), nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=self.ksizes[1]),
            nn.Dropout(dropout_rate), nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=self.ksizes[2]),
            nn.Dropout(dropout_rate), nn.ReLU(),
        )
        # self.convs = nn.ModuleList([nn.Conv2d(1, 100, (w, 200)) for w in kernel_wins])
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(128, num_class)

    def forward(self, x, meta):
      # print('x',x.shape,'meta',meta.shape)
      x = self.embedding(x) # [4, len, 300] (4=bsz)
      x = torch.transpose(x,1,2)
      x = self.conv_unit(x) # x1: [4, 128, len_a]
      x = x.squeeze(-1) # x: [4, 128, len_d]
      x = self.dropout(x) # torch.Size([4, 128, len_d])
      x = x[:,:,-1] # [bsz, 128]

      logit = self.fc(x) # [4, 6]
      return logit


class liar_dataset(Dataset):
  def __init__(self, dst, dummy_name):
    self.embedded = np.array(dst['embedded'])
    self.label = np.array(dst['label'])
    self.meta = np.array(dst[dummy_name])
  def __getitem__(self, index):
    return self.embedded[index],\
          self.label[index],\
          self.meta[index]
  def __len__(self):
    return len(self.label)


class CNN_model(nn.Module):
    def __init__(self, token_size, pretrained_emb, hidden_dim=64, n_layers=2):
        super(CNN_model, self).__init__()
        num_class = 6
        dropout_rate = 0.5
        self.ksizes = [3,4,5]
        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=300
        )
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

      
        self.conv_unit1 = nn.Sequential(
            torch.nn.Conv1d(in_channels=300, out_channels=128, kernel_size=self.ksizes[0]),
            torch.nn.MaxPool1d(kernel_size=self.ksizes[0]),
            # torch.nn.AdaptiveMaxPool1d(output_size),
        )
        self.conv_unit2 = nn.Sequential(
            torch.nn.Conv1d(in_channels=300, out_channels=128, kernel_size=self.ksizes[1]),
            torch.nn.MaxPool1d(kernel_size=self.ksizes[1]),
        )
        self.conv_unit3 = nn.Sequential(
            torch.nn.Conv1d(in_channels=300, out_channels=128, kernel_size=self.ksizes[2]),
            torch.nn.MaxPool1d(kernel_size=self.ksizes[2]),
        )
        # self.convs = nn.ModuleList([nn.Conv2d(1, 100, (w, 200)) for w in kernel_wins])
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(128, num_class)

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn = nn.LSTM(300, hidden_dim, num_layers=n_layers, bidirectional=True, dropout=0.5)
        self.dropout = nn.Dropout(0.8)

        self.meta_lstm = nn.LSTM(input_size=300, hidden_size = 5, num_layers = 2, 
                                 batch_first = True, bidirectional = True)
        self.meta_lstm1 = nn.LSTM(input_size=35, hidden_size = 16, num_layers = 2, 
                                 batch_first = True, bidirectional = True)

    def forward(self, x, meta):
      # print('x',x.shape,'meta',meta.shape)
      x = self.embedding(x) # [4, len, 300] (4=bsz)
      x = torch.transpose(x,1,2)
      x1 = self.conv_unit1(x) # x1: [4, 128, len_a]
      x2 = self.conv_unit2(x) # x2: [4, 128, len_b]
      x3 = self.conv_unit3(x) # x3: [4, 128, len_c]
      x = torch.cat((x1,x2,x3), dim=2) # x: [4, 128, len_d]
      x = x.squeeze(-1) # x: [4, 128, len_d]
      x = self.dropout(x) # torch.Size([4, 128, len_d])
      x = x[:,:,-1] # [bsz, 128]

      logit = self.fc(x) # [4, 6]
      return logit
      
class BiLSTM_Attention(nn.Module):
    def __init__(self, token_size, pretrained_emb, hidden_dim=64, n_layers=2):
        super(BiLSTM_Attention, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=300
        )
        print('hidden_dim',hidden_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb).type(torch.float))

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn = nn.LSTM(300, hidden_dim, num_layers=n_layers, bidirectional=True, dropout=0.5)
        self.dropout = nn.Dropout(0.5)

        self.w_omega = nn.Parameter(torch.Tensor(hidden_dim * 2, hidden_dim * 2))
        self.u_omega = nn.Parameter(torch.Tensor(hidden_dim * 2, 1))

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)
        self.fc = nn.Linear(hidden_dim * 2, 6)
        

    def attention_net(self, x):       #x:[batch, seq_len, hidden_dim*2]
        u = torch.tanh(torch.matmul(x, self.w_omega))         #[batch, seq_len, hidden_dim*2]
        att = torch.matmul(u, self.u_omega)                   #[batch, seq_len, 1]
        att_score = F.softmax(att, dim=1)
        scored_x = x * att_score                              #[batch, seq_len, hidden_dim*2]
        context = torch.sum(scored_x, dim=1)                  #[batch, hidden_dim*2]
        # print('context',context.shape)
        return context

    def forward(self, x, meta):
        # print(x.shape,meta.shape)
        embedding = self.dropout(self.embedding(x))       #[seq_len, batch, embedding_dim]
        embedding = torch.transpose(embedding,0,1)
        # output: [seq_len, batch, hidden_dim*2]     hidden/cell: [n_layers*2, batch, hidden_dim]
        output, (final_hidden_state, final_cell_state) = self.rnn(embedding)
        output = output.permute(1, 0, 2)                  #[batch, seq_len, hidden_dim*2]
        attn_output = self.attention_net(output)
        
        logit = self.fc(attn_output)
        return logit

        fc1 = nn.Linear(meta.shape[1],16)
        meta = fc1(meta.float())
        # print(meta.shape)

        output2 = torch.cat((attn_output,meta), dim=1)
        fc = nn.Linear(output2.shape[1], 6)
        output3 = fc(output2)
        return output3


def clean_text(w):
    return re.sub(
            r"([.,'!?\"()*#:;])",
            '',
            w.lower()
            ).replace('-', ' ').replace('/', ' ')
def preprocessing(train=False, eval=False, test=False):
  # label col:  "pants-fire" : 0, "false" : 1, "barely-true" : 2, "half-true" : 3, "mostly-true" : 4, "true" : 5
  if train:
    current_dataset = load_dataset("liar", split="train")
  if eval:
    current_dataset = load_dataset('liar', split='validation')
  if test:
    current_dataset = load_dataset('liar', split='test')
  return current_dataset

def get_word2vec_embedding(statements, data_dir):
  token_file = os.path.join(data_dir,'token_to_ix.pkl')
  w2v_file = os.path.join(data_dir,'train_w2v.npy')

  if os.path.exists(w2v_file) and os.path.exists(token_file):
        print("Loading train language files")
        return pickle.load(open(token_file, "rb")), np.load(w2v_file)

  token2ix = {'PAD': 0, 'UNK': 1}
  for s in statements:
    s = clean_text(s).split()
    for word in s:
      if word not in token2ix:
        token2ix[word] = len(token2ix)
  ix2token = {token2ix[k]: k for k in token2ix.keys()}
  w2v_path = '/content/gdrive/MyDrive/530project/GoogleNews-vectors-negative300.bin.gz'
  w2vmodel = gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=True)
  pretrained_emb = torch.randn([len(token2ix),300])
  for i in range(len(token2ix)):
    word = ix2token[i]
    if word in w2vmodel:
      vec = w2vmodel[word]
      pretrained_emb[i, :] = torch.from_numpy(vec)
  np.save(w2v_file, pretrained_emb)
  pickle.dump(token2ix, open(token_file, "wb"))
  return token2ix, pretrained_emb

def embed_text(x, max_len, token2ix):
  ques_ix = np.zeros(max_len, np.int64)
  x = clean_text(x).split()
  for ix, word in enumerate(x):
    if word in token2ix:
      ques_ix[ix] = token2ix[word]
    else:
      ques_ix[ix] = 1
    if ix + 1 == max_len:
      break
  return ques_ix
def category_from_output(output):
  res = []
  for i in output:
    top_n, top_i = i.topk(1)
    category_i = top_i[0].item()
    res.append(category_i)
  return res
def process_col(current_dataset, train = False, col_cnts = None, col = 'speaker'):
  if train:
    col_cnt = Counter(current_dataset[col])
    col_cnt = sorted(col_cnt.items(), key = lambda kv: kv[1], reverse=True)
    # elif col == 'speaker':
    col_cnt = {j[0]:idx for idx, j in enumerate([i for i in col_cnt if i[1]>60])}
  else:
    col_cnt = col_cnts[col]
  
  def col2ix(x):
    if x in col_cnt:
      return col_cnt[x]
    return len(col_cnt.keys())
  current_dataset[col + '_'] = current_dataset[col].apply(lambda x: col2ix(x))
  dummies = pd.get_dummies(current_dataset[col+'_'], prefix=col)
  names = list(dummies.columns)
  current_dataset = pd.concat((current_dataset,dummies),axis = 1)
  return current_dataset, names, col_cnt

def process_metadata(current_dataset, meta_cols, train = False, col_cnts = None):
  dummy_name = []
  if train: col_cnts = {}
  for col in meta_cols:
    current_dataset, names, col_cnt = process_col(current_dataset, train = train, col_cnts = col_cnts, col = col)
    dummy_name += names
    if train: col_cnts[col] = col_cnt
  return current_dataset, dummy_name, col_cnts
  
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def evaluate_model(model, data_iter, device, criteon):
  model.eval()
  y_pred, y_true = [], []
  for batch_idx, (text, label, meta) in enumerate(data_iter):
        # if text.shape[0]!=BATCH_SIZE:
          # continue
        text, label, meta = text.to(device), label.to(device), meta.to(device)
        output = model(text, meta)
        categories = category_from_output(output)
        loss = criteon(output,label)

        y_pred += categories
        y_true += label.tolist()
  # mf1 = f1_score(y_pred,y_true,average='macro')
  acc = accuracy_score(y_pred,y_true)
  # print('acc: ', acc,'Micro f1',mf1)
  print('acc: ', acc)
  
  