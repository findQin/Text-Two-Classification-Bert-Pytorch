import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from tqdm import tqdm
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)

class MyExample(object):

    def __init__(self,
                 book_id,
                 context_sentence,             
                 label=None):

        self.book_id = book_id
        self.context_sentence = context_sentence
        self.label = label



class InputFeatures(object):

    def __init__(self,
                 unique_id,
                 book_id,
                 input_ids,
                 input_mask,
                 label=None
                 ):
                 
        self.unique_id = unique_id
        self.book_id = book_id 
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.label = label



def get_split(text):
    l_total = []
    l_parcial = []
    
    # 双斜杠整除
    if len(text)//150 >0:
        n = len(text)//150
    else: 
        n = 1
        
    for w in range(n):
        if w == 0:
            l_parcial = text[:200]
            l_total.append("".join(l_parcial))
            
        else:
            l_parcial = text[w*150 : w*150 + 200]
            l_total.append("".join(l_parcial))
            
    return l_total



def read_data(data_path):
    
    df = pd.read_csv( data_path )

    df['review_seperate'] = df['review'].apply(get_split)

    text_l = []  # 分割好的文本
    label_l = []  # 每段文本的label
    index_l =[]   # 该段文本属于未分割前的哪个文本

    for _, row in df.iterrows():
        for l in row['review_seperate']:
            text_l.append(l)
            label_l.append(row['label'])
            index_l.append(row['id'])

    assert len(text_l) == len(label_l) 
    assert len(text_l) == len(index_l)

    for t in text_l:
        if len(t) > 510:
            print("---a data of text_l exceed 510!----")
    
    return text_l, label_l, index_l



def read_examples(text_l, label_l, index_l):
    examples = []
    
    # list 是有序的
    for i in range( len(text_l) ):
        examples.append(
            MyExample(
                book_id = index_l[i],
                context_sentence = text_l[i],
                label = label_l[i]
            )
        )
    
    return examples



def convert_examples_to_features(examples, tokenizer, max_seq_length=512):
    
    features = []
    unique_id = 1000000000 
    
    for _, example in enumerate( tqdm( examples, ncols=80, desc = "Make Data") ):
        context_tokens = tokenizer.tokenize(example.context_sentence) # comment
        
        tokens = []
        
        tokens.append("[CLS]")
        
        for c_token in context_tokens:
            tokens.append(c_token)
        
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        
        label = example.label
        
        features.append(
                InputFeatures(
                    unique_id=unique_id,
                    book_id=example.book_id,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    label=label
                )
            )
        
        unique_id += 1
    
    return features



def convert_features_to_tensors(features, batch_size, train):
    
    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    
    all_input_mask = torch.tensor(
        [f.input_mask for f in features], dtype=torch.long)
    
    all_label_ids = torch.tensor(
            [f.label for f in features], dtype=torch.long)
    
    if train:  
        data = TensorDataset(all_input_ids, all_input_mask,
                         all_label_ids)
                         
        sampler = RandomSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
               
    else:
        #all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long) 

        data = TensorDataset(all_input_ids, all_input_mask,
                         all_label_ids)
                         
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    
    return dataloader



def load_data(data_path, batch_size, train=False):

    bert_vocab_file = "model/bert/vocab.txt"

    tokenizer = BertTokenizer( vocab_file =  bert_vocab_file )

    text_l, label_l, index_l = read_data( data_path )

    examples = read_examples( text_l, label_l, index_l )

    features = convert_examples_to_features( examples, tokenizer )

    dataloader = convert_features_to_tensors( features, batch_size, train)

    return dataloader, len(features)



if __name__ == "__main__":    
    print(123456)
