
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from sklearn.metrics import accuracy_score, f1_score

output_dir = '../output'
def evaluate(y_pred, y_test):
    acc = accuracy_score(y_pred, y_test)
    mf1 = f1_score(y_pred, y_test,average='macro')
    print('acc: ', acc,'Micro f1',mf1)

def plot_confusion_matrix(y_pred, y_test):
    # y_pred, y_test = val_pred, val_true
    cm = confusion_matrix(y_test, y_pred)
    for i in range(len(cm)): # only show error
      cm[i][i]=0
    # label_dict = {"false" : 0, "half-true" : 1, "mostly-true" : 2, "true": 3, "barely-true" : 4, "pants-fire" : 5 } 
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

    labels = ["false", "half-true", "mostly-true", "true", "barely-true", "pants-fire"]
    df = pd.DataFrame(cm, columns=labels, index= labels)
    df.index.name = "True Label"
    df.columns.name = "Predicted Label"
    # plt.title('XLNet')
    plt.figure(figsize=(7,7))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    ax = sns.heatmap(df, annot = True, cmap=cmap,cbar=False,fmt='g',)
    ax.set_xlabel('Predicted Label',fontsize = 15)
    ax.set_ylabel('True Label',fontsize = 15)
    ax.set_title('', fontsize=10)
    for tick in ax.xaxis.get_majorticklabels():  # example for xaxis
            tick.set_fontsize(15) 
    for tick in ax.yaxis.get_majorticklabels():  # example for xaxis
            tick.set_fontsize(15) 
    # ax.legend(fontsize=15)
    plt.tight_layout()
    plt.show()

if __name__ =="__main__":
    y_pred = pd.read_csv(os.path.join(output_dir, 'y_pred.csv'))['y']
    y_true = pd.read_csv(os.path.join(output_dir, 'y_true.csv'))['y']
    evaluate(y_pred, y_true)
    plot_confusion_matrix(y_pred, y_true)




