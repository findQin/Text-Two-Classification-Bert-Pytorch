from transformers.modeling_bert import BertModel

import torch
from torch import nn
from torch.nn import CrossEntropyLoss



class BertOrigin(nn.Module):
    """BERT model for multiple choice tasks. BERT + Linear

    Args:
        config: BertConfig 类对象， 以此创建模型
        num_choices: 选项数目，默认为 2.
    """

    def __init__(self, config, num_classes):
    
        super().__init__()
        
        self.num_classes = num_classes
        
        self.bert = BertModel.from_pretrained( config.bert_model_dir )
        
        for param in self.bert.parameters():
            param.requires_grad = True 
        
        self.dropout = nn.Dropout( config.hidden_dropout_prob )
        
        self.classifier = nn.Linear( config.hidden_size, num_classes )
        
    
    def forward(self, input_ids, attention_mask):
        """
        Inputs:
            input_ids: [batch_size, num_choices, sequence_length]， 其中包含了词所对应的ids
            attention_mask: 可选，[batch_size, num_choices, sequence_length]；区分 padding 与 token， 1表示是token，0 为padding
            labels: [batch_size], 其中数字在 [0, ..., num_choices]之间
        """
        
        batch_size = input_ids.shape[0]

        bert_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=False)
         
        # model.get_sequence_output() == bert_output[0]: (batch_size, sequence_length, hidden_size)
        
        # 获取每个token的output 输出[batch_size, seq_length, embedding_size]
        
        # model.get_sequence_output()获取每个单词的词向量的时候注意，头尾是[CLS]和[SEP]的向量。
        
        # bert_output[1]: (batch_size, hidden_size) 是获取这句句子的向量output，一个768维的向量，这个向量是具有上下文信息的
        
        pooled_output = bert_output[1]

        pooled_output = self.dropout(pooled_output)
                
        logits = self.classifier(pooled_output).view(batch_size, self.num_classes)

        #dim=-1表示按行计算
        logits = nn.functional.softmax(logits, dim=-1)
        # logits: (batch_size, num_classes)
        
        return logits