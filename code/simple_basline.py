# data_dir = '../data'
data_dir = data_dir = '../data'
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Times New Roman']
sns.set_style("whitegrid")
sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})
sns.set(font_scale=1.2)

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
  current_dataset['label_'] = current_dataset['label'].apply(lambda x: get_label(x))
  current_dataset.reset_index(drop=True,inplace=True)
  return current_dataset
train_dataset = preprocessing(train=True)
dev_dataset = preprocessing(eval=True)
test_dataset = preprocessing(test=True)

# dev_dataset['label']
lab = ["true", "mostly-true", "half-true", "barely-true", "false","pants-fire"]
lab_dict = {}
# for l in lab:
lab_dict = {l:len(train_dataset[train_dataset['label']==l]) for l in lab}
print(lab_dict)
lab_dict = sorted(lab_dict.items(), key = lambda kv: kv[1], reverse=True)
majority = lab_dict[0][0]
print('majority label:',majority)

from sklearn.metrics import accuracy_score, f1_score

def eval(dst,lab,cur = 'Dev set'):
    print('Accuracy on ',cur,len(dst[dst['label']==lab])/len(dst))
    y_pred = [lab for i in range(len(dst))]
    mf1 = f1_score(y_pred,dst['label'],average='macro')
    print('Macro F1 score on ',cur,mf1)

eval(dev_dataset,majority)
eval(test_dataset,majority,cur = 'Test set')
