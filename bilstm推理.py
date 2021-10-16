import torch.nn.functional as F # pytorch 激活函数的类
from torch import nn,optim # 构建模型和优化器
from collections import defaultdict
from operator import itemgetter
import numpy as np
import torch
import torch.nn.functional as F # pytorch 激活函数的类
import pickle as pk
import pandas as pd
from tqdm import tqdm
from model_zoo import *


# 构建基于bilstm实现ner
class bilstm(nn.Module):
    def __init__(self, parameter):
        super(bilstm, self).__init__()
        word_size = parameter['word_size']
        embedding_dim = parameter['d_model']
        # 此处直接基于id，对字进行编码
        self.embedding = nn.Embedding(word_size, embedding_dim, padding_idx=0)

        hidden_size = parameter['hid_dim']
        num_layers = parameter['n_layers']
        dropout = parameter['dropout']
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, bidirectional=True, batch_first=True, dropout=dropout)

        output_size = parameter['output_size']
        self.fc = nn.Linear(hidden_size*2, output_size)
        
        
    def forward(self, x):
        out = self.embedding(x)
        out,(h, c)= self.lstm(out)
        out = self.fc(out)
        return out.view(-1,out.size(-1))  #将所有句子的字放到一个维度上拍平
    

def load_model(model_name):
    
    parameter = pk.load(open('parameter.pkl', 'rb'))
    if 'bert' in model_name:
        model = eval(model_name+"(config,parameter).to(parameter['device'])")
    else:
        model = eval(model_name+"(parameter).to(parameter['device'])")
    model.load_state_dict(torch.load(model_name+'.h5'))
    model.eval()
    return model, parameter

def list2torch(ins):
    return torch.from_numpy(np.array(ins))

def batch_yield(parameter, shuffle = True, isTrain = True, isBert = False ):
    data_set = parameter['data_set']['train'] if isTrain else parameter['data_set']['dev']
    Epoch = parameter['epoch'] if isTrain else 1
    parameter['batch_size'] = 10
    for epoch in range(Epoch):
        if shuffle:
            random.shuffle(data_set)
        inputs, targets = [], []
        max_len = 0
        for items in tqdm(data_set):
            input = itemgetter(*items[0])(parameter['word2ind'])
            input = input if type(input) == type(()) else (input,0)
            target = itemgetter(*items[1])(parameter['key2ind'])
            target = target if type(target) == type(()) else (target,0)
            if len(input)>max_len:
                max_len = len(input)
            inputs.append(list(input))
            targets.append(list(target))
            if len(inputs) >= parameter['batch_size']:
               inputs = [i + [0]*(max_len - len(i)) for i in inputs] 
               targets = [i + [-1]*(max_len - len(i)) for i in targets]
               yield list2torch(inputs),list2torch(targets), None, False
               inputs, targets = [], []
               max_len = 0
        inputs = [i + [0]*(max_len-len(i)) for i in inputs]
        targets = [i + [-1]*(max_len-len(i)) for i in targets]
        yield list2torch(inputs),list2torch(targets), epoch, False
        inputs, targets = [], []
        max_len =0
    yield None, None, None, True

def eval_model(model_name):
    model, parameter = load_model(model_name)
    count_table = {}
    test_yield = batch_yield(parameter, shuffle=False, isTrain=False)
    while 1:
        
        inputs, targets, _, keys = next(test_yield)
        if not keys:
            pred = model(inputs.long().to(parameter['device']))
            predicted_prob, predicted_index = torch.max(F.softmax(pred,1),1)
            predicted_index = predicted_index.reshape(inputs.shape)
            targets = targets.long().to(parameter['device'])
            right = (targets == predicted_index)
            for i in range(1, parameter['output_size']):
                if i not in count_table:
                    count_table[i] = {'pred':len(predicted_index[(predicted_index == i)]),
                                      'real':len(targets[targets == i]),
                                      'common':len(targets[right & (targets == i)])}
                else:
                    count_table[i]['pred'] += len(predicted_index[predicted_index == i])
                    count_table[i]['real'] += len(targets[targets == i])
                    count_table[i]['common'] += len(targets[right & (targets == i)])
        else:
            break
    
    count_pandas = {}
    name,count = list(parameter['key2ind'].keys())[1:], list(count_table.values())
    for ind, i in enumerate(name):
        i = i.split('-')[1]
        if i in count_pandas:
            count_pandas[i][0] += count[ind]['pred']
            count_pandas[i][1] += count[ind]['real']
            count_pandas[i][2] += count[ind]['common']
        else:
            count_pandas[i] = [0,0,0]
            count_pandas[i][0] = count[ind]['pred']
            count_pandas[i][1] = count[ind]['real']
            count_pandas[i][2] = count[ind]['common']
    #计算所有标签下的pred总数，real的总数，和common的总数        
    count_pandas['all'] = [sum([count_pandas[i][0] for i in count_pandas]),
                           sum([count_pandas[i][1] for i in count_pandas]),
                           sum([count_pandas[i][2] for i in count_pandas])]
    
    name = count_pandas.keys()
    count_pandas = pd.DataFrame(count_pandas.values())
    count_pandas.columns = ['pred','real','common']
    count_pandas['p'] = count_pandas['common']/count_pandas['pred']
    count_pandas['r'] = count_pandas['common']/count_pandas['real']
    count_pandas['f1'] = 2 * count_pandas['p']*count_pandas['r']/(count_pandas['p']+count_pandas['r'])
    count_pandas.index = list(name)
    return count_pandas
eval_model('bilstm')

def keyword_predict(input):
    input = list(input)
    input_id = tokenizer.convert_tokens_to_ids(input)
    predict = model.crf.decode(model(list2torch([input_id]).long().to(parameter['device'])))[0]
    predict = itemgetter(*predict)(parameter['ind2key'])
    keys_list = []
    for ind,i in enumerate(predict):
        if i == 'O':
            continue
        if i[0] =='S':
            if not(len(keys_list) == 0 or keys_list[-1][-1]):
                del keys_list[-1]
            keys_list.append([input[ind],[i],[ind],True])
            continue
        if i[0] == 'B':
            if not(len(keys_list) == 0 or keys_list[-1][-1]):
                del keys_list[-1]
            keys_list.append([input[ind],[i],[ind],False])
            continue
        if i[0] == 'I':
            if len(keys_list)>0 and not keys_list[-1][-1] and \
            keys_list[-1][1][0].split('-')[1] == i.split('-')[1]:
                keys_list[-1][0] += input[ind]
                keys_list[-1][1] += [i]
                keys_list[-1][2] += [ind]
            else:
                if len(keys_list)>0:
                    del keys_list[-1]
            continue
        if i[0] == 'E':
            if len(keys_list)>0 and not keys_list[-1][-1] and \
            keys_list[-1][1][0].split('-')[1] == i.split('-')[1]:
                keys_list[-1][0] += input[ind]
                keys_list[-1][1] += [i]
                keys_list[-1][2] += [ind]
                keys_list[-1][3] = True
            else:
                if len(keys_list)>0:
                    del keys_list[-1]
            continue
    return keys_list
model, parameter = load_model('bert_crf')
tokenizer = tokenizer_class.from_pretrained('prev_trained_model')
model = model.to(parameter['device'])
test_text = '浙商银行企业信贷部叶老桂博士则从另一个角度对五道门槛进行了解读。叶老桂认为，对目前国内商业银行而言'
print(keyword_predict(test_text))
                    
                
            
    
               
        
               
