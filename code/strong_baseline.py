# from google.colab import drive
# drive.mount('/content/gdrive')

# !pip install datasets



from strong_baseline_helper import *

data_dir = '.'
model_path = os.path.join(data_dir, 'liar_CNN.pth')

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
        # if text.shape[0]!=BATCH_SIZE:
          # continue
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
    args = argparser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    EMBEDDING_DIM = 300
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    EPOCH = 11

    train_dataset = pd.DataFrame(preprocessing(train=True))
    statements = train_dataset['statement']
    token2ix, pretrained_emb = get_word2vec_embedding(statements, data_dir)
    print(pretrained_emb.shape) # (len(vocab), embedding_dim)
    lengths = [len(x.split()) for x in statements]
    max_len = int(np.percentile(lengths,90))

    train_dataset['embedded'] = train_dataset['statement'].apply(lambda x: embed_text(x, max_len, token2ix))
    dev_dataset = pd.DataFrame(preprocessing(eval=True))
    dev_dataset['embedded'] = dev_dataset['statement'].apply(lambda x: embed_text(x, max_len, token2ix))
    test_dataset = pd.DataFrame(preprocessing(test=True))
    test_dataset['embedded'] = test_dataset['statement'].apply(lambda x: embed_text(x, max_len, token2ix))

    meta_cols = ['subject','speaker','job_title','state_info','party_affiliation','context']
    train_dataset_meta, dummy_name, col_cnts = process_metadata(train_dataset, meta_cols, train = True, col_cnts = None)
    dev_dataset_meta, _, _ = process_metadata(dev_dataset,meta_cols, train = False, col_cnts = col_cnts)
    test_dataset_meta, _, _ = process_metadata(test_dataset,meta_cols, train = False, col_cnts = col_cnts)

    BATCH_SIZE = 64
    train_dst = liar_dataset(train_dataset_meta, dummy_name)
    train_data_iter = DataLoader(train_dst, batch_size=BATCH_SIZE, shuffle=True)
    dev_dst = liar_dataset(dev_dataset_meta, dummy_name)
    dev_data_iter = DataLoader(dev_dst, batch_size=BATCH_SIZE, shuffle=True)
    test_dst = liar_dataset(test_dataset_meta, dummy_name)
    test_data_iter = DataLoader(test_dst, batch_size=BATCH_SIZE, shuffle=True)


    net = CNN_model(len(token2ix), pretrained_emb).to(device)
    criteon = nn.CrossEntropyLoss().to(device)
    opt = optimizer.Adam(net.parameters(), lr=LEARNING_RATE)
    if args.train:
        train_losses, dev_losses, dev_acc_list, best_model = train(11,train_data_iter,dev_data_iter,opt,criteon, net, device)
        torch.save(best_model.state_dict(), os.path.join(data_dir,'liar_CNN.pth'))
    if args.eval:
        best_model = CNN_model(len(token2ix), pretrained_emb).to(device)
        best_model.load_state_dict(torch.load(model_path, map_location='cpu'))
        evaluate_model(best_model, dev_data_iter, device, criteon)
    if args.test:
        best_model = CNN_model(len(token2ix), pretrained_emb).to(device)
        best_model.load_state_dict(torch.load(model_path, map_location='cpu'))
        evaluate_model(best_model, test_data_iter, device, criteon)
    if args.eval or args.test:
        X_train, y_train = np.array([np.array(i) for i in train_dataset['embedded']]),np.array(train_dataset['label'])
        X_dev, y_dev = np.array([np.array(i) for i in dev_dataset['embedded']]),np.array(dev_dataset['label'])
        X_test, y_test = np.array([np.array(i) for i in test_dataset['embedded']]),np.array(test_dataset['label'])
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        models = {'logistic':LogisticRegression(),'svm':SVC()}
        for m in models:
          print(m)
          model = models[m]
          model.fit(X_train,y_train)
          y_pred = model.predict(X_dev)
          print('dev',accuracy_score(y_pred,y_dev))
          # print('dev f1',f1_score(y_pred,y_dev,average='macro'))
          
          y_pred = model.predict(X_test)
          print('test',accuracy_score(y_pred,y_test))
          # print('test f1',f1_score(y_pred,y_test,average='macro'))