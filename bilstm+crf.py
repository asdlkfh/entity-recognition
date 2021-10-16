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
    'epoch':100,
    # 单次训练的batch_size为100条数据
    'batch_size':100,
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
    data_set = {}                     #各句话中的字，和每个字对应的标签
    key_table = defaultdict(int)      #每个标签和其对应的频率
    vocab_table = defaultdict(int)    #字：字频
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
                vocab_table[j] += 1    #将文本中的所有的字和字频放入到该字典中
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

def batch_yield(parameter,shuffle = True,isTrain = True):
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
            input = itemgetter(*items[0])(parameter['word2ind'])  #取出每句话中所有字的索引
            target = itemgetter(*items[1])(parameter['key2ind'])  #将句子中的实体所属类型的索引拿出来
            input = input if type(input) == type(()) else (input,0)
            target = target if type(target) == type(()) else (target,0)
            if len(input) > max_len:
                max_len = len(input)
            inputs.append(list(input))
            targets.append(list(target))
            if len(inputs) >= parameter['batch_size']:
                # 填空补齐
                inputs = [i+[0]*(max_len-len(i)) for i in inputs]  #不足max_len的部分用0 pading
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
    return torch.from_numpy(np.array(ins))

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
    parameter = pk.load(open('parameter.pkl','rb'))   #将parameter_copy里的参数添加到parameter里面
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
    
    
import torch.nn.functional as F # pytorch 激活函数的类
from torch import nn,optim # 构建模型和优化器

# 构建基于bilstm实现ner
#class bilstm(nn.Module):
    #def __init__(self, parameter):
        #super(bilstm, self).__init__()
        #word_size = parameter['word_size']
        #embedding_dim = parameter['d_model']
        ## 此处直接基于id，对字进行编码
        #self.embedding = nn.Embedding(word_size, embedding_dim, padding_idx=0)

        #hidden_size = parameter['hid_dim']
        #num_layers = parameter['n_layers']
        #dropout = parameter['dropout']
        #self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, bidirectional=True, batch_first=True, dropout=dropout)

        #output_size = parameter['output_size']
        #self.fc = nn.Linear(hidden_size*2, output_size)
        
        
    #def forward(self, x):
        #out = self.embedding(x)
        #out,(h, c)= self.lstm(out)
        #out = self.fc(out)
        #return out.view(-1,out.size(-1))  #将所有句子的字放到一个维度上拍平
    
    
#import os
#import shutil
#import pickle as pk
#from torch.utils.tensorboard import SummaryWriter


## 构建模型
#model = bilstm(parameter).to(parameter['device'])    #实例初始化模型，然后传入cuda

## 确定训练模式
#model.train()

## 确定优化器和损失
#optimizer = torch.optim.SGD(model.parameters(),lr=0.1, momentum=0.95, nesterov=True)
#criterion = nn.CrossEntropyLoss()

## 准备迭代器
#train_yield = batch_yield(parameter)

## 开始训练
#loss_cal = []
#min_loss = float('inf')
#while 1:
        #inputs,targets,epoch,keys = next(train_yield)
        #if keys:
            #break
        #out = model(inputs.long().to(parameter['device']))
        #loss = criterion(out, targets.view(-1).long().to(parameter['device']))
        #optimizer.zero_grad()
        #loss.backward()
        #optimizer.step()
        #loss_cal.append(loss.item())
        #if epoch is not None:
            #if (epoch+1)%1 == 0:
                #loss_cal = sum(loss_cal)/len(loss_cal)
                #if loss_cal < min_loss:
                    #min_loss = loss_cal
                    #torch.save(model.state_dict(), 'bilstm.h5')
                #print('epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, \
                                                       #parameter['epoch'],loss_cal))
            #loss_cal = [loss.item()]
            
            
import torch.nn.functional as F # pytorch 激活函数的类
from torch import nn,optim # 构建模型和优化器
from torchcrf import CRF


# 构建基于bilstm+crf实现ner
class bilstm_crf(nn.Module):
    def __init__(self, parameter):
        super(bilstm_crf, self).__init__()
        word_size = parameter['word_size']
        embedding_dim = parameter['d_model']
        self.embedding = nn.Embedding(word_size, embedding_dim, padding_idx=0)

        hidden_size = parameter['hid_dim']
        num_layers = parameter['n_layers']
        dropout = parameter['dropout']
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, bidirectional=True, batch_first=True, dropout=dropout)

        output_size = parameter['output_size']
        self.fc = nn.Linear(hidden_size*2, output_size)
        
        self.crf = CRF(output_size,batch_first=True)
        
    def forward(self, x):
        out = self.embedding(x)
        out,(h, c)= self.lstm(out)
        out = self.fc(out)
        return out
    
    
import os
import shutil
import pickle as pk
from torch.utils.tensorboard import SummaryWriter


# 构建模型
model = bilstm_crf(parameter).to(parameter['device'])

# 确定训练模式
model.train()

# 确定优化器和损失
optimizer = torch.optim.SGD(model.parameters(),lr=0.00005, momentum=0.95, nesterov=True)
# optimizer = torch.optim.Adam(model.parameters(),lr = parameter['lr'], \
#                              weight_decay = 0.01)

# 准备学习率策略
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

# 准备迭代器
train_yield = batch_yield(parameter)

# 开始训练
loss_cal = []
min_loss = float('inf')
while 1:
        inputs,targets,epoch,keys = next(train_yield)
        if keys:
            break
        out = model(inputs.long().to(parameter['device']))
        # crf被用于损失
        loss = -model.crf(out,targets.long().to(parameter['device']))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_cal.append(loss.item())
        if epoch is not None:
            if (epoch+1)%1 == 0:
                loss_cal = sum(loss_cal)/len(loss_cal)
                if loss_cal < min_loss:
                    min_loss = loss_cal
                    torch.save(model.state_dict(), 'bilstm_crf.h5')
                print('epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, \
                                                       parameter['epoch'],loss_cal))
            loss_cal = [loss.item()]
            scheduler.step()
