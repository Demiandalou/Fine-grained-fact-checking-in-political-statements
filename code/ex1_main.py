from ex1_helper import *

def evaluate_model(model, data_iter):
  model.eval()
  y_pred, y_true = [], []
  for batch_idx, (text, label, meta) in enumerate(data_iter):
        text, label, meta = text.to(device), label.to(device), meta.to(device)
        output = model(text, meta)
        categories = category_from_output(output)
        loss = criteon(output,label)

        y_pred += categories
        y_true += label.tolist()
        
  acc = accuracy_score(y_pred,y_true)#, f1_score(y_pred,y_true)
  # print('acc: ', acc)
  mf1 = f1_score(y_pred,y_true,average='macro')
  print('acc: ', acc,'Micro f1',mf1)
  return y_pred,y_true


def train(epoch,train_data_iter,dev_data_iter,opt,criteon, net, device):
  def timeSince(since):
      now = time.time()
      s = now - since
      m = math.floor(s / 60)
      s -= m * 60
      return '%dm %ds' % (m, s)
  train_losses, dev_losses, dev_acc_list = [], [], []
  best_model, best_val_acc = None, float('-inf')
  cnt_step = 0
  current_loss = 0
  plot_every = 2
  dev_every = 2
  print('train len:',len(train_data_iter),'dev len:',len(dev_data_iter))
  print('learning_rate',LEARNING_RATE,'n_iters',epoch, 'batch size', BATCH_SIZE, 'optim','Adam', 'lr_scheduler',None, 'device',device)
  start = time.time()
  for e in range(epoch): 
    print('Epoch', e)
    net.train()
    for batch_idx, (text, label, meta) in enumerate(train_data_iter):
      # if text.shape[0]!=BATCH_SIZE:
        # continue
      text, label, meta = text.to(device), label.to(device), meta.to(device)
      output = net(text,meta)
      loss = criteon(output,label)
      current_loss += loss
      cnt_step += 1
      opt.zero_grad()
      loss.backward()
      opt.step()
    if e==0:
      print(time.time()-start)
    if e % plot_every == 0:
      tmp_loss = current_loss.item() / cnt_step
      train_losses.append(tmp_loss)
      current_loss, cnt_step = 0, 0
      print('%d %d%% (%s) %.4f ' % (e, e / EPOCH * 100, timeSince(start), tmp_loss))
    if e % dev_every ==0:
      net.eval()
      eval_loss = 0
      y_pred, y_true = [], []
      cnt_eval_step = 0
      for batch_idx, (text, label, meta) in enumerate(dev_data_iter):
        text, label, meta = text.to(device), label.to(device), meta.to(device)
        output = net(text, meta)
        categories = category_from_output(output)
        loss = criteon(output,label)
        eval_loss += loss
        cnt_eval_step += 1

        y_pred += categories
        y_true += label.tolist()
      # print(cnt_eval_step, eval_loss, len(dev_data_iter))
      dev_losses.append(eval_loss.item()/cnt_eval_step)
      acc = accuracy_score(y_pred,y_true)
      dev_acc_list.append(acc)
      if acc>best_val_acc:
        best_val_acc = acc
        best_model = copy.deepcopy(net)
      print('%d %d%% (%s) %.4f %s %s %.4f' % (e, e / EPOCH * 100, timeSince(start), eval_loss.item()/cnt_eval_step, categories[:4], label.tolist()[:4], acc))
  print('best_val_acc',best_val_acc)
  return train_losses, dev_losses, dev_acc_list, best_model # best_model

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='530 milestone2')
    argparser.add_argument('--train', default=0, action='store_true')
    argparser.add_argument('--eval', default=1, action='store_true')
    argparser.add_argument('--test', default=1, action='store_true')
    argparser.add_argument('--save_testy', default=1, action='store_true')
    args = argparser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    EMBEDDING_DIM = 300
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    EPOCH = 11
    model_path = os.path.join(data_dir,'liar_BiLSTM_Attention-acc2850-2762.pth')
    # model_path = os.path.join(data_dir,'liar_BiLSTM_Attention-acc2952-2636.pth')
    
    
    train_dataset = preprocessing(train=True)
    statements = train_dataset['statement']
    token2ix, pretrained_emb = get_word2vec_embedding(statements, data_dir)
    print(pretrained_emb.shape) # (len(vocab), embedding_dim)
    lengths = [len(x.split()) for x in statements if type(x)!=float]
    max_len = int(np.percentile(lengths,90))

    dev_dataset = preprocessing(eval=True)
    test_dataset = preprocessing(test=True)

    token2ix_meta, pretrained_emb_meta, max_len_meta = get_meta_embed(train_dataset, meta_cols)

    train_dst = liar_dataset(train_dataset,max_len,token2ix, token2ix_meta, max_len_meta)
    train_data_iter = DataLoader(train_dst, batch_size=BATCH_SIZE, shuffle=True)
    dev_dst = liar_dataset(dev_dataset ,max_len,token2ix, token2ix_meta, max_len_meta)
    dev_data_iter = DataLoader(dev_dst, batch_size=BATCH_SIZE, shuffle=True)
    test_dst = liar_dataset(test_dataset,max_len,token2ix, token2ix_meta,max_len_meta)
    test_data_iter = DataLoader(test_dst, batch_size=BATCH_SIZE, shuffle=True)


    LEARNING_RATE = 0.0002
    criteon = nn.CrossEntropyLoss().to(device)
    net = BiLSTM_Attention(len(token2ix), pretrained_emb, len(token2ix_meta), pretrained_emb_meta, hidden_dim=48, n_layers = 2, dropout = 0.3).to(device)
    opt = optimizer.Adam(net.parameters(), lr=LEARNING_RATE,weight_decay=1e-4)
    if args.train:
        train_losses, dev_losses, dev_acc_list, best_model = train(21,train_data_iter,dev_data_iter,opt,criteon, net, device)
        torch.save(best_model.state_dict(), os.path.join(data_dir,'liar_biLSTM.pth'))
    if args.eval or args.test:
        best_model = BiLSTM_Attention(len(token2ix), pretrained_emb, len(token2ix_meta), pretrained_emb_meta, hidden_dim=48, n_layers = 2, dropout = 0.3).to(device)
        best_model.load_state_dict(torch.load(model_path))
        print('dev')
        val_pred,val_true = evaluate_model(best_model, dev_data_iter) # 0.2850467289719626
        print('test')
        test_pred,test_true = evaluate_model(best_model, test_data_iter) # 0.27624309392265195

        if args.save_testy:
            df = pd.DataFrame({'y':test_pred})
            df.to_csv(os.path.join(output_dir,'y_pred.csv'))
            df = pd.DataFrame({'y':test_true})
            df.to_csv(os.path.join(output_dir,'y_true.csv'))

