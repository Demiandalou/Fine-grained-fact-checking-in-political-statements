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

data_dir = '../data'
output_dir = '../output'

meta_cols = ['subject','speaker','job_title','state_info','party_affiliation','context','justification']
class liar_dataset(Dataset):
  def __init__(self, dst,max_len,token2ix, token2ix_meta, max_len_meta):
    self.embed_statement = np.array([embed_text(i,max_len,token2ix) for i in list(dst['statement'])])
    self.label = np.array(dst['label'])
    self.embed_meta = np.array(embed_meta(dst,meta_cols, token2ix_meta, max_len_meta))
  def __getitem__(self, index):
    return self.embed_statement[index],\
          self.label[index],\
          self.embed_meta[index]
  def __len__(self):
    return len(self.label)

class BiLSTM_Attention(nn.Module): # 2850-2762 hidden_dim=48, n_layers = 2, dropout = 0.3
    def __init__(self, token_size, pretrained_emb, token_size_meta, pretrained_emb_meta, 
                 hidden_dim=64, n_layers=2,dropout = 0.5):
        super(BiLSTM_Attention, self).__init__()
        print('hidden_dim',hidden_dim, 'n_layers',n_layers, 'dropout',dropout)

        self.embedding = nn.Embedding(num_embeddings=token_size,
                                      embedding_dim=300)
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb).type(torch.float))

        self.meta_embedding = nn.Embedding(num_embeddings=token_size_meta,
                                      embedding_dim=300)
        self.meta_embedding.weight.data.copy_(torch.from_numpy(pretrained_emb_meta).type(torch.float))

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn = nn.LSTM(300, hidden_dim, num_layers=n_layers, bidirectional=True, dropout=dropout)
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

        meta = self.meta_embedding(meta)
        embedding = torch.cat((embedding,meta), dim = 1)

        embedding = torch.transpose(embedding,0,1)
        output, (final_hidden_state, final_cell_state) = self.rnn(embedding) # [28, 64, 128]
        output = output.permute(1, 0, 2)                  #[batch, seq_len, hidden_dim*2]
        attn_output = self.attention_net(output) # [64, 128]
        logit = self.fc(attn_output)
        return logit

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

def get_word2vec_embedding(statements, data_dir):
  token_file = os.path.join(data_dir,'token_to_ix_w2v.pkl')
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

def get_glove_embedding(reviews, data_dir):
  token_file = os.path.join(data_dir,'token_to_ix_glove.pkl')
  glove_file = os.path.join(data_dir,'train_glove.npy')
  if os.path.exists(glove_file) and os.path.exists(token_file):
        print("Loading saved embedding")
        return pickle.load(open(token_file, "rb")), np.load(glove_file)
  all_reviews = {}
  for idx, s in enumerate(reviews):
    all_reviews[idx] = clean_text(s).split()

  from collections import defaultdict
  token_to_ix = defaultdict(int)
  token_to_ix['UNK'] = 0
  token_to_ix['SS'] = 1

  spacy_tool = en_vectors_web_lg.load()
  pretrained_emb = []
  pretrained_emb.append(spacy_tool('UNK').vector)
  pretrained_emb.append(spacy_tool('SS').vector)
  
  for k, v in all_reviews.items():
      for word in v:
          if word not in token_to_ix:
              token_to_ix[word] = len(token_to_ix)
              pretrained_emb.append(spacy_tool(word).vector)

  pretrained_emb = np.array(pretrained_emb)
  np.save(glove_file, pretrained_emb)
  pickle.dump(token_to_ix, open(token_file, "wb"))
  return token_to_ix, pretrained_emb

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
  token2ix, pretrained_emb = get_glove_embedding(all_text, data_dir)
  lengths = [len(x.split()) for x in all_text]
  max_len = int(np.percentile(lengths,90))
  return token2ix,pretrained_emb, max_len
def embed_meta(dst, meta_cols, token2ix, max_len):
  all_features, all_text = [], []
  for i in range(len(dst)):
    cur = ''
    for c in meta_cols:
      try:
        cur += str(dst[c][i]) + ' SS '
      except:
        print(c,i)
        return
    all_text.append(cur)
  for t in all_text:
    all_features.append(embed_text(t, max_len, token2ix))
  return np.array(all_features)
