
from ex2_helper import *

def get_dataloader(text, metad, label):
    input_ids,token_type_ids,attention_mask = [],[],[]
    input_ids_meta,token_type_ids_meta,attention_mask_meta = [],[],[]
    labels = []
    for i,t in enumerate(text):
        if str(t)=='nan':
          t = ' '
        encoded = tokenizer.encode_plus(text=t,max_length=max_len,padding='max_length',truncation=True)
        # print(encoded)
        input_ids.append(encoded['input_ids'])
        # token_type_ids.append(encoded['token_type_ids'])
        attention_mask.append(encoded['attention_mask'])
        labels.append(int(label[i]))

        encoded_mata = tokenizer.encode_plus(text=metad[i],max_length=max_len_meta,padding='max_length',truncation=True)
        input_ids_meta.append(encoded_mata['input_ids'])
        # token_type_ids_meta.append(encoded_mata['token_type_ids'])
        attention_mask_meta.append(encoded_mata['attention_mask'])

    input_ids,attention_mask = torch.tensor(input_ids),torch.tensor(attention_mask)
    labels = torch.tensor(labels)
    input_ids_meta,attention_mask_meta = torch.tensor(input_ids_meta),torch.tensor(attention_mask_meta)
    data = TensorDataset(input_ids,attention_mask,labels,\
        input_ids_meta,attention_mask_meta)
    loader = DataLoader(data,batch_size=BATCH_SIZE,shuffle=True) 
    return loader

def train(model,train_loader,valid_loader,optimizer,schedule,device,epoch, name = 'liar_xlnet.pth', curtype = 'meta'):
    train_losses, dev_losses, dev_acc_list = [],[],[]
    print('device',device,'epoch',epoch,'train',len(train_loader),\
          'val',len(valid_loader), 'batch', BATCH_SIZE,'lr',LEARNING_RATE)
    best_acc = 0.0
    criterion = nn.CrossEntropyLoss()  
    for i in range(epoch):
        start = time.time()
        model.train()
        print("### Epoch {} ###".format(i+1))
        train_loss_sum = 0.0
        for idx,(ids,att,y,idm,attm) in enumerate(train_loader):
            ids,att,y,idm,attm = ids.to(device),att.to(device),y.to(device),idm.to(device),attm.to(device)
            if curtype=='meta':
              ids = torch.concat((ids,idm),dim=1)
              att = torch.concat((att,attm),dim=1)
            y_pred = model(ids) 
            loss = criterion(y_pred,y) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            schedule.step()
            train_loss_sum += loss.item()
            
            if(idx+1)%(len(train_loader)//5)==0:
                print("epoch {:04d}, step {:04d}/{:04d}, loss {:.4f}, time {:.4f}".format(
                i+1,idx+1,len(train_loader),train_loss_sum/(idx+1),time.time()-start))
                # break
        train_losses.append(train_loss_sum/len(train_loader))

        model.eval()
        acc, dev_loss = evaluate(model,valid_loader,device, curtype = curtype)  
        dev_losses.append(dev_loss)
        dev_acc_list.append(acc)
        if acc > best_acc:
            best_acc = acc
            best_model = copy.deepcopy(model)
            # torch.save(model.state_dict(),"best_roberta_model.pth") 
            torch.save(model.state_dict(), os.path.join(data_dir,name))
        print("current acc is {:.4f},best acc is {:.4f}".format(acc,best_acc))
        print("time costed = {}s \n".format(round(time.time()-start,5)))
    return train_losses, dev_losses, dev_acc_list, best_model
   
def evaluate(model,data_loader,device,name='cloth_bert.pth', curtype='meta'):
    criterion = nn.CrossEntropyLoss()  
    model.eval()
    val_true,val_pred = [],[]
    all_loss=0
    for idx,(ids,att,y,idm,attm) in enumerate(data_loader):
        # print(device)
        ids,att,y,idm,attm = ids.to(device),att.to(device),y.to(device),idm.to(device),attm.to(device)
        if curtype=='meta':
          ids = torch.concat((ids,idm),dim=1)
          att = torch.concat((att,attm),dim=1)
        y_pred = model(ids)
        loss = criterion(y_pred,y) 
        all_loss+=loss.item()

        categories = category_from_output(y_pred)
        val_pred += categories
        val_true += y.tolist()
        if idx==0:
          print(val_pred,val_true)
    acc = accuracy_score(val_pred,val_true)
    mf1 = f1_score(val_pred,val_true,average='macro')
    print('acc: ', acc,'Micro f1',mf1)
    return acc, all_loss/len(data_loader)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='530 milestone2')
    argparser.add_argument('--train', default=0, action='store_true')
    argparser.add_argument('--eval', default=1, action='store_true')
    argparser.add_argument('--test', default=1, action='store_true')
    args = argparser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    max_len = 28
    max_len_meta = 132
    BATCH_SIZE = 16
    EPOCHS = 11
    LEARNING_RATE = 0.00015
    tokenizer = XLNetTokenizer.from_pretrained(bert_dir)
    model = XLNetModel.from_pretrained(bert_dir)

    train_dataset = preprocessing(train=True)
    dev_dataset = preprocessing(eval=True)
    test_dataset = preprocessing(test=True)


    train_loader = get_dataloader(train_dataset['statement'], get_meta_embed(train_dataset, meta_cols), label = train_dataset['label'])
    dev_loader = get_dataloader(dev_dataset['statement'], get_meta_embed(dev_dataset, meta_cols), label = dev_dataset['label'])
    test_loader = get_dataloader(test_dataset['statement'], get_meta_embed(test_dataset, meta_cols), label = test_dataset['label'])

    
    if args.train:
        model = XLNet_Model(bert_dir).to(device)
        optimizer = AdamW(model.parameters(),lr=LEARNING_RATE,weight_decay=1e-4)

        schedule = get_cosine_schedule_with_warmup(optimizer,num_warmup_steps=len(train_loader),num_training_steps=EPOCHS*len(test_loader))
        train_losses, dev_losses, dev_acc_list,best_model = train(model,train_loader,dev_loader,optimizer,schedule,device,15)

        import warnings
        warnings.filterwarnings("ignore")
        fig, ((ax1, ax2))= plt.subplots(1,2,figsize = (15,5))
        x_axis = [i for i in range(len(dev_losses))]
        sns.lineplot(x_axis, dev_acc_list, ax = ax1)
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel("Number of Iterations")
        sns.lineplot(x_axis, train_losses, ax = ax2, label = 'train loss')
        sns.lineplot(x_axis, dev_losses, ax = ax2, label = 'dev loss')
        ax2.set_ylabel("Loss")
        ax2.set_xlabel("Number of Iterations")
        ax2.legend()
        plt.tight_layout()
        plt.show()
    
    if args.eval or args.test:
        model = XLNet_Model(bert_dir).to(device)
        model.load_state_dict(torch.load(os.path.join('liar_xlnet-2780-2794.pth')))
        evaluate(model, dev_loader,device)
        evaluate(model, test_loader,device)
