from transformers.optimization import (
    AdamW, get_linear_schedule_with_warmup, get_constant_schedule)

from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)
from tqdm import tqdm
from copy import deepcopy
from collections import OrderedDict

from BertOrigin import args
from utils.shopping_utils import load_data
from utils.util import get_device, acc_and_f1

import numpy as np
import torch
import torch.nn as nn
import os



def train(epoch_num, 
        n_gpu, 
        train_dataloader,
        dev_dataloader, 
        valid_train_dataloader,
        model, 
        optimizer, 
        criterion, 
        gradient_accumulation_steps, 
        max_grad_norm, 
        device, 
        scheduler, 
        output_model_path):
        
    best_model_state_dict, best_dev_f1, global_step = None, 0, 0  
        
    for epoch in range( int( epoch_num ) ):
        
        model.train()

        print(f'-------------- Epoch: {epoch+1:02} ----------')

        for step, batch in enumerate( tqdm( train_dataloader, desc = "Iteration" ) ):
            
            batch = tuple(t.to(device) for t in batch)      

            input_ids, input_mask, label_ids = batch

            print(input_ids.shape)
            print(input_mask.shape)
            print(label_ids.shape)

            logits = model( input_ids, input_mask )

            loss = criterion(logits, label_ids)

            if n_gpu > 1:
                loss = loss.mean()
            
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            
            optimizer.zero_grad()

            # 反向传播
            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_grad_norm)
                
                # 更新参数                
                optimizer.step()
                # 更新学习率
                scheduler.step()               
                global_step += 1
            
        print('---traing is Ok----')  

        train_acc, train_f1 = evaluate(
            model, valid_train_dataloader, device)


        dev_acc, dev_f1 = evaluate(
            model, dev_dataloader,  device)
        
        s = 'Epoch: {:d}, train_acc: {:.6f}, train_f1: {:.6f}, ' \
            'valid_acc: {:.6f}, valid_f1: {:.6f}, '.format(
            epoch+1, train_acc, train_f1, dev_acc, dev_f1)

        print(s)

        with open(os.path.join(output_model_path,'save_run_result.txt'), 'a', encoding = 'utf-8') as f:
            f.write(s)
            f.write('\n')

        # torch.save(model.state_dict(), os.path.join(
        #     output_model_path, 'bert-' + str(epoch + 1) + '.bin'))

        if dev_f1 > best_dev_f1: 
            best_model_state_dict = deepcopy(model.state_dict())
            best_dev_f1 = dev_f1

    return best_model_state_dict                                    



def evaluate(model, data_loader, device):

    preds = None
    out_label_ids = None
    model.eval()

    for batch in tqdm(data_loader, desc="Evaluating", ncols=80):
        
        batch = tuple(t.to(device) for t in batch)

        input_ids, input_mask, label_ids = batch

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=input_mask)
            
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = label_ids.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, label_ids.detach().cpu().numpy(), axis=0)

    preds = np.argmax(preds, axis=1)

    results = acc_and_f1(preds, out_label_ids)

    return results




def main(config, bert_vocab_file, do_prediction=False):

    if not os.path.exists( config.output_dir ):
        os.makedirs( config.output_dir )
        
    # --gpu_ids: [1,2,3]--
    gpu_ids = [int(device_id) for device_id in config.gpu_ids.split(',')] 
    print("gpu_ids:{}".format(gpu_ids))
    
    device, n_gpu = get_device(gpu_ids[0])

    if n_gpu > 1:
        n_gpu = len(gpu_ids)
           
    #label_list = ["0", "1"]

    criterion = nn.CrossEntropyLoss()   
    criterion = criterion.to(device)

    if not do_prediction:
        # 数据准备
        train_file = os.path.join(config.data_dir, "train.csv")   
        dev_file = os.path.join(config.data_dir, "valid.csv")

        train_dataloader, train_len = load_data(train_file, config.batch_size, train=True)
        print("Num train_set: {}".format(train_len))
        
        valid_train_dataloader, valid_train_len= load_data(train_file, config.batch_size)
        print("Num valid_train_set: {}".format(valid_train_len))

        dev_dataloader, dev_len = load_data(dev_file, config.batch_size)
        print("Num dev_set: {}".format(dev_len))
        
        num_train_steps = int(
            train_len / config.batch_size / config.gradient_accumulation_steps * config.num_train_epochs)
        
        if config.model_name == "BertOrigin":
            from BertOrigin.BertOrigin import BertOrigin 
            model = BertOrigin(config, num_classes = 2)  
        
        model.to(device)   
        if n_gpu > 1:
            model = nn.DataParallel(model, device_ids=gpu_ids)
            
        no_decay = ['bias', 'gamma', 'beta']
            
        optimizer_parameters = [
            {'params': [p for n, p in model.named_parameters()
                        if not any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.01},
            
            {'params': [p for n, p in model.named_parameters()
                        if any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.0}]
                
        optimizer = AdamW(
            optimizer_parameters,
            lr = config.learning_rate,
            betas = (0.9, 0.999),
            weight_decay = 1e-8,
            correct_bias = False)

        # bert里的小技巧, bert里的learning rate是不断变化的,先往上升再往下降,这个scheduler就是用来设置这个      
        scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps = config.num_warmup_steps,
                num_training_steps = num_train_steps)

        best_model_state_dict = train(config.num_train_epochs, n_gpu, train_dataloader, dev_dataloader, valid_train_dataloader, model, optimizer, criterion,
                                    config.gradient_accumulation_steps, config.max_grad_norm, device, scheduler, config.output_dir)

        torch.save(best_model_state_dict, config.best_model_file)
    
    else:
        print('---**Enter Test**---')
        
        #dev_dataloader, dev_examples, dev_features, dev_labels = dev[:-1]

        test_file = os.path.join(config.data_dir, "test.csv")   
        test_dataloader, test_len = load_data(
            test_file, config.batch_size)
        
        print('Num test_set: {}'.format(test_len))
        
        if config.model_name == "BertOrigin":
            from BertOrigin.BertOrigin import BertOrigin 
            test_model = BertOrigin(config, num_classes = 2)
        
        
        pretrained_model_dict = torch.load(config.best_model_file)
        new_state_dict = OrderedDict()
        for k, value in pretrained_model_dict.items():
            #name = k[7:] # remove `module.`
            new_state_dict[k] = value        

        test_model.load_state_dict(new_state_dict, strict=True)
        test_model.to(device)

        if n_gpu > 1:
            test_model = nn.DataParallel(test_model, device_ids=gpu_ids)

        test_acc, test_f1 = evaluate(
            test_model, test_dataloader, device)
        
        print(f'\t  Acc: {test_acc*100: .3f}% | f1: {test_f1*100: .3f}%')    



if __name__ == "__main__":

    model_name = "BertOrigin"   
    data_dir = "dataset/"   
    gpu_ids = "3"
    
    # roberta
    bert_vocab_file = "model/bert/vocab.txt"
    
    # do_prediction: False(训练), True(预测)
    main(args.get_args(data_dir,gpu_ids), bert_vocab_file, do_prediction=True)      
    