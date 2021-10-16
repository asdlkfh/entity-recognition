from collections import defaultdict
from operator import itemgetter
from tqdm import tqdm
import numpy as np
import random
import torch 
import jieba
import json
import os

import pickle as pk

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.set_device(0)
else:
    device = torch.device('cpu')
# 确定模型训练方式，GPU训练或CPU训练
parameter_copy = {
    # 此处embedding维度为768
    'd_model':768, 
    # rnn的隐层维度为300
    'hid_dim':300,
    # 训练的批次为100轮
    'epoch':5,
    # 单次训练的batch_size为100条数据
    'batch_size':50,
    # 设置两个lstm，原文应该是一个
    'n_layers':2,
    # 设置dropout，为防止过拟合
    'dropout':0.1,
    # 配置cpu、gpu
    'device':device,
    # 设置训练学习率
    'lr':0.001,
    # 优化器的参数，动量主要用于随机梯度下降
    'momentum':0.99,
}

def build_dataSet(parameter):
    data_name = ['train','dev']
    # 准备相应的字典
    data_set = {}
    key_table = defaultdict(int)
    vocab_table = defaultdict(int)
    # 预先准备相应的标志位
    vocab_table['<PAD>'] = 0
    vocab_table['<UNK>'] = 0
    # 数据内容可以参考data文件夹下的README，基于CLUENER 数据进行处理
    # 因为有两份数据，dev和train，因为构建时候同时进行构建
    for i in data_name:
        data_set[i] = []
        data_src = open('data/'+i+'.json','r',encoding = 'utf-8').readlines()
        for data in data_src:
            # 加载相应的数据
            data = json.loads(data)
            # 获取对应的文本和标签
            text = list(data['text'])
            label = data['label']
            # 初始化标准ner标签
            label_new = ['O']*len(text)
            key_table['O']
            # 根据其所带有的标签，如game、address进行数据提取
            for keys in label:
                inds = label[keys].values()
                # 因为其标签下的数据是一个数组，代表这类型标签的数据有多个
                # 因此循环处理，且护士其keys（文本内容），因为可以通过id索引到
                for id_list in inds:
                    for ind in id_list:
                        if ind[1] - ind[0] == 0:
                            # 当id号相同，表明这个实体只有一个字，
                            # 那么他的标签为'S-'+对应的字段
                            keys_list = ['S-'+keys]
                            label_new[ind[0]] = keys_list[0]
                        if ind[1] - ind[0] == 1:
                            # 如果id号相差，仅为1，表明这个实体有两个字
                            # 那么他的标签为 B-*，E-*，表明开始和结束的位置
                            keys_list = ['B-'+keys,'E-'+keys]
                            label_new[ind[0]] = keys_list[0]
                            label_new[ind[1]] = keys_list[1]
                        if ind[1] - ind[0] > 1:
                            # 如果id号相差，大于1，表明这个实体有多个字
                            # 那么他的标签除了 B-*，E-*，表明开始和结束的位置
                            # 还应该有I-*，来表明中间的位置
                            keys_list = ['B-'+keys,'I-'+keys,'E-'+keys]
                            label_new[ind[0]] = keys_list[0]
                            label_new[ind[0]+1:ind[1]] = [keys_list[1]]*(ind[1]-1-ind[0])
                            label_new[ind[1]] = keys_list[2]
                        for key in keys_list:
                            # 为了后面标签转id，提前准好相应的字典
                            key_table[key] += 1
            # 此处用于构建文本的字典
            for j in text:
                vocab_table[j] += 1
            # 保存原始的文本和处理好的标签
            data_set[i].append([text,label_new])
    # 保存标签转id，id转标签的字典
    key2ind = dict(zip(key_table.keys(),range(len(key_table))))
    ind2key = dict(zip(range(len(key_table)),key_table.keys()))
    # 保存字转id，id转字的字典
    word2ind = dict(zip(vocab_table.keys(),range(len(vocab_table))))
    ind2word = dict(zip(range(len(vocab_table)),vocab_table.keys()))
    parameter['key2ind'] = key2ind
    parameter['ind2key'] = ind2key
    parameter['word2ind'] = word2ind
    parameter['ind2word'] = ind2word
    parameter['data_set'] = data_set
    parameter['output_size'] = len(key2ind)
    parameter['word_size'] = len(word2ind)
    return parameter


def batch_yield_bert(parameter,shuffle = True,isTrain = True):
    # 构建数据迭代器
    # 根据训练状态或非训练状态获取相应数据
    data_set = parameter['data_set']['train'] if isTrain else parameter['data_set']['dev']
    Epoch = parameter['epoch'] if isTrain else 1
    for epoch in range(Epoch):
        # 每轮对原始数据进行随机化
        if shuffle:
            random.shuffle(data_set)
        inputs,targets = [],[]
        max_len = 0
        for items in tqdm(data_set):
            # 基于所构建的字典，将原始文本转成id，进行多分类
            # 此处和bilstm处不一致，使用bert自带字典
            input = tokenizer.convert_tokens_to_ids(items[0])
            target = itemgetter(*items[1])(parameter['key2ind'])
            target = target if type(target) == type(()) else (target,0)
            if len(input) > max_len:
                max_len = len(input)
            inputs.append(list(input))
            targets.append(list(target))
            if len(inputs) >= parameter['batch_size']:
                # 填空补齐
                inputs = [i+[0]*(max_len-len(i)) for i in inputs]
                targets = [i+[0]*(max_len-len(i)) for i in targets]
                yield list2torch(inputs),list2torch(targets),None,False
                inputs,targets = [],[]
                max_len = 0
        inputs = [i+[0]*(max_len-len(i)) for i in inputs]
        targets = [i+[0]*(max_len-len(i)) for i in targets]
        yield list2torch(inputs),list2torch(targets),epoch,False
        inputs,targets = [],[]
        max_len = 0
    yield None,None,None,True
            

def list2torch(ins):
    return torch.from_numpy(np.array(ins)).long().to(parameter['device'])

# 因此这边提前配置好用于训练的相关参数
# 不要每次重新生成
if not os.path.exists('parameter.pkl'):
    parameter = parameter_copy
    # 构建相关字典和对应的数据集
    parameter = build_dataSet(parameter)
    pk.dump(parameter,open('parameter.pkl','wb'))
else:
    # 读取已经处理好的parameter，但是考虑到模型训练的参数会发生变化，
    # 因此此处对于parameter中模型训练参数进行替换
    parameter = pk.load(open('parameter.pkl','rb'))
    for i in parameter_copy.keys():
        if i not in parameter:
            parameter[i] = parameter_copy[i]
            continue
        if parameter_copy[i] != parameter[i]:
            parameter[i] = parameter_copy[i]
    for i in parameter_copy.keys():
        print(i,':',parameter[i])
    pk.dump(parameter,open('parameter.pkl','wb'))
    del parameter_copy,i
    

#基于bert预训练模型
# 加载transformers相关用于模型训练的包
from transformers import WEIGHTS_NAME, BertConfig,get_linear_schedule_with_warmup,AdamW, BertTokenizer
from transformers import BertModel,BertPreTrainedModel
from torch.nn import CrossEntropyLoss
import torch.nn as nn
import torch

import torch.nn.functional as F # pytorch 激活函数的类
from torch import nn,optim # 构建模型和优化器

class bert(BertPreTrainedModel):
    def __init__(self, config,parameter):
        super(bert, self).__init__(config)
        self.num_labels = config.num_labels
        # 写法就是torch的写法，区别在于BertModel的网络结构已经不需要自己完成
        # 上游特征提取
        self.bert = BertModel(config)                         
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 下游对应任务处理，直接使用fc，进行线性变换即可
        embedding_dim = parameter['d_model']
        output_size = parameter['output_size']
        self.fc = nn.Linear(embedding_dim, output_size)
        self.init_weights()
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None,labels=None):
        # 基于bert进行特征提取（上游）
        outputs = self.bert(input_ids = input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        # 提取对应的encoder输出
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        # 下游直接进行全连接
        logits = self.fc(sequence_output)
        # 完成模型输出
        return logits.view(-1,logits.size(-1))
    
# 加载bert自带的config文件，初始化bert需要
# 加载预训练模型bert的字典，数据处理时需要使用
config_class, bert, tokenizer_class = BertConfig, bert, BertTokenizer
config = config_class.from_pretrained("prev_trained_model")             #取出文件中的参数
tokenizer = tokenizer_class.from_pretrained("prev_trained_model")


import os
import shutil
import pickle as pk
from torch.utils.tensorboard import SummaryWriter

random.seed(2019)

# 构建模型，并加载预训练权重
model = bert.from_pretrained("prev_trained_model",config=config,parameter = parameter).to(parameter['device'])#将权重和参数一块传入类中，去初始化模型

# 确定训练权重，可以只训练fc层，但是还是推荐全部训练
full_finetuning = True  
if full_finetuning:     #全部训练
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 
             'weight_decay': 0.01},                                                                     #所有不在no_decay里的参数
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 
             'weight_decay': 0.0}      #该方法是为了不同的参数用不同的方法优化                              #在no_decay里的参数
        ]
else: 
        param_optimizer = list(model.fc.named_parameters())    #只训练最后的fc层
        optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer]}]

# 确定训练的优化器和学习策略，这个是bert常用的优化方法和学习策略，可以还是基于以前的方式，只是效果会有所下降
optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5, correct_bias=False)
train_steps_per_epoch = 10748 // parameter['batch_size']
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=train_steps_per_epoch, num_training_steps=parameter['epoch'] * train_steps_per_epoch)


# 确定训练模式
model.train()

# 确定损失
criterion = nn.CrossEntropyLoss(ignore_index=-1)


# 准备迭代器
train_yield = batch_yield_bert(parameter)

# 开始训练
loss_cal = []
min_loss = float('inf')
logging_steps = 0
while 1:
        inputs,targets,epoch,keys = next(train_yield)
        if keys:
            break
        out = model(inputs)
        loss = criterion(out, targets.view(-1))
        optimizer.zero_grad()
        loss.backward()
        # 适当梯度修饰
        nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=5)
        # 优化器和学习策略更新
        optimizer.step()
        scheduler.step()
        
        loss_cal.append(loss.item())
        logging_steps += 1
        if logging_steps%100 == 0:
            print(sum(loss_cal)/len(loss_cal))
        if epoch is not None:
            if (epoch+1)%1 == 0:
                loss_cal = sum(loss_cal)/len(loss_cal)
                if loss_cal < min_loss:
                    min_loss = loss_cal
                    torch.save(model.state_dict(), 'bert.h5')
                print('epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, \
                                                       parameter['epoch'],loss_cal))
            loss_cal = [loss.item()]
            scheduler.step()




#基于bert预训练模型+CRF
from transformers import WEIGHTS_NAME, BertConfig,get_linear_schedule_with_warmup,AdamW, BertTokenizer
from transformers import BertModel,BertPreTrainedModel
from torch.nn import CrossEntropyLoss
import torch.nn as nn
import torch
from torchcrf import CRF

import torch.nn.functional as F # pytorch 激活函数的类
from torch import nn,optim # 构建模型和优化器

# 方法与bert没有什么区别，只是加上了CRF进行处理
# 构建基于bert+crf实现ner
class bert_crf(BertPreTrainedModel):   #继承BertPreTrainedModel,则需要将config从外部传入；#继承nn.Module,则需要将config在里边传入
    def __init__(self, config,parameter):
        super(bert_crf, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        embedding_dim = parameter['d_model']
        output_size = parameter['output_size']
        self.fc = nn.Linear(embedding_dim, output_size)
        self.init_weights()
        
        self.crf = CRF(output_size,batch_first=True)
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None,labels=None):
        outputs = self.bert(input_ids = input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.fc(sequence_output)
        return logits
    
config_class, bert_crf, tokenizer_class = BertConfig, bert_crf, BertTokenizer
config = config_class.from_pretrained("prev_trained_model")
tokenizer = tokenizer_class.from_pretrained("prev_trained_model")


import os
import shutil
import pickle as pk
from torch.utils.tensorboard import SummaryWriter

random.seed(2019)

# 构建模型
model = bert_crf.from_pretrained("prev_trained_model",config=config,parameter = parameter).to(parameter['device'])

# 确定训练权重
full_finetuning = True
if full_finetuning:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 
             'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 
             'weight_decay': 0.0}
        ]
else: 
        param_optimizer = list(model.fc.named_parameters()) 
        optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer]}]

# 确定训练的优化器和学习策略
optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5, correct_bias=False)
train_steps_per_epoch = 10748 // parameter['batch_size']
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=train_steps_per_epoch, num_training_steps=parameter['epoch'] * train_steps_per_epoch)


# 确定训练模式
model.train()

# 确定损失
criterion = nn.CrossEntropyLoss(ignore_index=-1)


# 准备迭代器
train_yield = batch_yield_bert(parameter)

# 开始训练
loss_cal = []
min_loss = float('inf')
logging_steps = 0
while 1:
        inputs,targets,epoch,keys = next(train_yield)
        if keys:
            break
        out = model(inputs)
        # 同样crf被用于损失
        loss = -model.crf(out, targets)
        optimizer.zero_grad()
        loss.backward()
        # 适当梯度修饰
        nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=5)
        # 优化器和学习策略更新
        optimizer.step()
        scheduler.step()
        
        loss_cal.append(loss.item())
        logging_steps += 1
        if logging_steps%100 == 0:
            print(sum(loss_cal)/len(loss_cal))
        if epoch is not None:
            if (epoch+1)%1 == 0:
                loss_cal = sum(loss_cal)/len(loss_cal)
                if loss_cal < min_loss:
                    min_loss = loss_cal
                    torch.save(model.state_dict(), 'bert_crf.h5')
                print('epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, \
                                                       parameter['epoch'],loss_cal))
            loss_cal = [loss.item()]


