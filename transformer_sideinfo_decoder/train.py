# !/usr/bin/env python
# coding: utf-8

# In[1]:

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import time
import math
import os
import sys

from torch.nn.init import xavier_uniform_

sys.path.append("/home/xumingshi/Minimize_Portfolio_Risk/transformer/src/")
print(os.getcwd())
print(os.path.abspath('.'))
plt.rcParams['font.sans-serif'] = [u'SimHei']
plt.rcParams['axes.unicode_minus'] = False
from preprocess import *
from dataloader import *

torch.manual_seed(0)
np.random.seed(0)
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.set_device(0)

# In[2]:


batch_size = 256 # batch size
num_asset = 32
feature_size = int(num_asset*(num_asset+1)/2) # 协方差矩阵的上三角部分组成的向量

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# device = ("cpu")


# In[3]:

'''
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        # pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]
'''
class PositionalEncoding(nn.Embedding):
    def __init__(self,d_model, dropout = 0.1,max_len = 5000):
        super().__init__(max_len, d_model)
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x):
        # print("positional encoding")
        weight = self.weight.data.unsqueeze(1)
        x = x + weight[:x.size(0),:]
        return self.dropout(x)


class LinearSideInformationEncoding(nn.Embedding):
    def __init__(self,d_model, dropout = 0.1,max_len = 5000):
        super().__init__(max_len, d_model)
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x, side_info):
        # print("positional encoding")
        #for i in range(side_info.size(1)):
        for j in range(side_info.size(2)):
            # print(side_info.shape)
            # print(x.shape)
            Linear_weight = self.weight.data.unsqueeze(1).repeat(1,side_info.size(1),1)
            # Quar_weight = self.weight.data.unsqueeze(1).repeat(1,side_info.size(1),1)
            # print(Linear_weight.shape)
            # print(Quar_weight.shape)
            # print(side_info.size(1))
            # print(side_info[:,:,j].shape)
            s = side_info[:,:,j].reshape(x.size(0),x.size(1),1)
            # torch.Size([batch_size, input_length, feature_size])+torch.Size([batch_size, input_length, feature_size])*torch.Size([batch_size, input_length, 1])
            # +torch.Size([batch_size, input_length, feature_size])*torch.Size([batch_size, input_length, 1])
            x = x + torch.mul(Linear_weight[:x.size(0),:],s) # + torch.mul(Quar_weight[:x.size(0),:],(s**2))
        return self.dropout(x)



class QuadraticSideInformationEncoding(nn.Embedding):
    def __init__(self,d_model, dropout = 0.1,max_len = 5000):
        super().__init__(max_len, d_model)
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x, side_info):
        # print("positional encoding")
        #for i in range(side_info.size(1)):
        for j in range(side_info.size(2)):

            Quadratic_weight = self.weight.data.unsqueeze(1).repeat(1,side_info.size(1),1)
            s = side_info[:,:,j].reshape(x.size(0),x.size(1),1)
            # torch.Size([batch_size, input_length, feature_size])+torch.Size([batch_size, input_length, feature_size])*torch.Size([batch_size, input_length, 1])
            x = x + torch.mul(Quadratic_weight[:x.size(0),:],(s**2))
        return self.dropout(x)

# In[4]:
'''
    def __init__(self, feature_size=feature_size, side_feature_size = 5, num_layers=3, dropout=0.1):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'

        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=8, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size + side_feature_size, feature_size)
        self.init_weights()   
'''
class TransAm(nn.Module):

    def __init__(self, feature_size=feature_size, side_feature_size = 5, num_layers=3, dropout=0.1):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        self.feature_size = feature_size
        self.side_feature_size = side_feature_size
        self.src_mask = None
        self.tgt_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.decode_pos_encoder = PositionalEncoding(feature_size)
        self.linear_side_info_encoder = LinearSideInformationEncoding(feature_size)
        self.quadratic_side_info_encoder = QuadraticSideInformationEncoding(feature_size)
        print("Encoder")
        # Encoder
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=8, dropout=dropout)
        self.encoder_norm = nn.LayerNorm(feature_size)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers, self.encoder_norm)
        print("Decoder")
        # Decoder
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=feature_size, nhead=8, dropout=dropout)
        self.decoder_norm = nn.LayerNorm(feature_size)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers, self.decoder_norm)
        print("Output")
        # Output
        self.output_layer = nn.Linear(feature_size, feature_size)
        self.reset_parameters()

    def reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, src, side_info):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            # print('a',src.size())
            mask = self._generate_square_subsequent_mask(len(src)).to(device)  # why?
            self.tgt_mask = mask
        src = self.pos_encoder(src)
        # print('j',src.size(),self.src_mask.size())
        src = self.linear_side_info_encoder(src, side_info)
        src = self.quadratic_side_info_encoder(src, side_info)
        memory = self.transformer_encoder(src)
        # print(memory.shape)
        # print(side_info.shape)
        # memory = torch.cat((memory, side_info), 2)
        # print(memory.shape)
        # tgt = torch.cat((src, side_info), 2)
        tgt = self.pos_encoder(src)
        output = self.transformer_decoder(tgt, memory, tgt_mask=self.tgt_mask)
        output = self.output_layer(output)

        return output

    def _generate_square_subsequent_mask(self,sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        # mask = mask.unsqueeze(0)
        # mask = mask.repeat(batch_size,1,1)
        return mask


# In[5]:
'''
for sparse type 0:
best parameter: 
    lag_t = 42
    input_gap = 21
    rebalance = 21
    input_length = 10
    output_length = 1
for sparse type 1:
best parameter: 
    lag_t = 21
    rebalance = 21
    input_length = 10
    output_length = 1
    lag_size = 1
for sparse type 2:
best parameter: 
    lag_t = 21
    rebalance = 21
    input_length = 10
    output_length = 1
    lag_size = 1
for sparse type 3:
best parameter: 
    lag_t = 21
    rebalance = 21
    input_length = 7
    output_length = 1
    lag_size = 5
'''
def get_preprocess_result():
    closes = np.load("/home/xumingshi/Minimize_Portfolio_Risk/transformer/data/close_price.npy")
    closes = np.log(closes)[1:] - np.log(closes)[:-1]
    # num_asset = 32
    lag_t = 42
    input_gap = 21
    rebalance = 42
    input_length = 30
    output_length = 1
    normal = True
    returns = closes[:, :num_asset]
    pick = np.arange(num_asset)
    cov_scale = "not_log"
    sparse_type = 0
    local_size = 5  # only for LocalAttention + LogSparse Self Attention
    lag_size = 5
    save_dir = '/home/xumingshi/Minimize_Portfolio_Risk/transformer/data/' + "num_%i_lag_%i_type_%i/" % (num_asset, lag_t, sparse_type)

    '''
    sparse_type:
    0: Linear Sparse Self Attention
    1: Log Sparse Self Attention
    2: LocalAttention + LogSparse Self Attention
    3: Log lagged Sparse Self Attention
    '''
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    np.save(save_dir + "pick.npy", pick)
    result = final_preprocess(returns, num_asset, lag_t, input_length=input_length, input_gap=input_gap, \
                              rebalance=rebalance, output_length=output_length, normal=normal, low=0, upper=1, \
                              cov_scale=cov_scale, sparse_type=sparse_type, local_size = local_size, lag_size = lag_size)
    return result, pick


def side_data():
    side_info = np.load("/home/xumingshi/Minimize_Portfolio_Risk/transformer/data/macro/combine.npy")
    print("loading side information, lenth, feature size, side_info shape: ", end = '')
    print(len(side_info), end=', ')
    print(len(side_info[0]), end=', ')
    side_feature = len(side_info[0])
    side_info = side_info.astype(float)
    side_info = torch.from_numpy(side_info).to(torch.float32)
    side_info = side_info.to(device)
    print(side_info.shape)
    return side_feature, side_info

# In[32]:

import tqdm

def train(train_dataset):
    model.train()  # Turn on the train mode

    total_loss = 0.
    start_time = time.time()
    print("train data")
    print("train dataset length: ", end='')
    print(len(train_dataset))
    side_feature_size, side_info = side_data()
    # for batch, i in enumerate(range(0, len(train_data) - 1, batch_size)):
    for i, (idx, inputVar, targetVar) in enumerate(train_data_loader):
        inputs = inputVar.to(device)  # B,S,C,H,W
        label = targetVar.to(device)  # B,S,C,H,W
        # print(inputs.shape) # torch.Size([batch_size, 7, feature_size])
        # print(label.shape) # torch.Size([batch_size, 1, feature_size])
        optimizer.zero_grad()
        idx = idx.long().to(device)
        # print(idx[-1])
        # print(inputs.shape)
        # print(idx.shape)
        # print(idx.dtype)
        # print(side_info[idx[:,-1]].shape)
        output = model(inputs, side_info[idx[:,:]])
        # print(output.shape)
        # print(label.shape)
        # if calculate_loss_over_all_values:
        loss = criterion(output[:,-1,:], label[:, -1, :])
        # else:
        #    loss = criterion(output[-output_window:], label[-output_window:])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = int(len(train_dataset) / batch_size / 3) # 2874/ batch_size / 5
        if batch_size % log_interval == 0 and batch_size > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | '
                  'lr {:02.6f} | {:5.2f} ms | '
                  'loss {:5.5f} | ppl {:8.2f}'.format(
                epoch, len(train_dataset) // batch_size, scheduler.get_lr()[0],
                              elapsed * 1000 / log_interval,
                cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
    print("train loss: ", end='')
    print(total_loss/len(train_dataset))


def uniform_evaluation(pred, label, feature_size = feature_size, maps = None):
    pred = pred.detach()
    label = label.detach()
    diff = pred - label
    diff_shape = diff.shape
    diff = torch.reshape(diff, (diff_shape[0] * diff_shape[1], -1))
    diff = diff[:, :feature_size]
    # print(diff.shape)
    square_diff = torch.sum(diff ** 2)
    avg_diff = square_diff / feature_size
    return avg_diff.item()


def evaluate(eval_model, data_source):
    eval_model.eval()  # Turn on the evaluation mode
    total_loss = 0.
    prev_uni_losses = 0
    test_uni_losses = 0
    # eval_batch_size = 1000
    print("evaluation")
    print("validation data_source length: ", end = '')
    print(len(data_source))
    side_feature_size, side_info = side_data()
    with torch.no_grad():
        '''
        for i in range(0, len(data_source) - 1, eval_batch_size):
            data, targets = get_batch(data_source, i, eval_batch_size)
        '''
        for i, (idx, inputVar, targetVar) in enumerate(valid_data_loader):
            inputs = inputVar.to(device)  # B,S,C,H,W
            label = targetVar.to(device)  # B,S,C,H,W
            idx = idx.long().to(device)
            output = eval_model(inputs, side_info[idx[:,:]])
            # print(output[-output_window:].size(), label[-output_window:].size())
            #if calculate_loss_over_all_values:
            loss = criterion(output[:,-1,:], label[:, -1, :])
            #else:
            #    loss = criterion(output[-output_window:], label[-output_window:])
            total_loss += loss.item()

            prev_loss = uniform_evaluation(inputs[:,-1,:], label[:,-1, :])
            uni_loss = uniform_evaluation(output[:,-1,:], label[:, -1, :])
            prev_uni_losses += prev_loss
            test_uni_losses += uni_loss

    print("validation loss: ", end='')
    print(total_loss / len(data_source))
    prev_uni_loss = prev_uni_losses / len(data_source)
    test_uni_loss = test_uni_losses / len(data_source)
    gain = 1 - (test_uni_loss / prev_uni_loss)
    print("validation gain: ", end='')
    print(gain)
    return total_loss / len(data_source)


def test(eval_model, data_source):
    print("test")
    print("test data_source length: ", end='')
    print(len(data_source))
    eval_model.eval()  # Turn on the evaluation mode
    test_uni_losses = 0
    prev_uni_losses = 0
    total_loss = 0
    side_feature_size, side_info = side_data()
    label_list = []
    output_list = []
    input_list = []

    for i, (idx, inputVar, targetVar) in enumerate(test_data_loader):
        ######################
        #   Test the model   #
        ######################
        inputs = inputVar.to(device)
        label = targetVar.to(device)
        idx = idx.long().to(device)
        output = eval_model(inputs, side_info[idx[:,:]])
        prev_loss = uniform_evaluation(inputs[:,-1,:], label[:, -1, :])
        uni_loss = uniform_evaluation(output[:,-1,:], label[:, -1, :])
        # if calculate_loss_over_all_values:
        loss = criterion(output[:,-1,:], label[:,-1,:])
        clone_label = label.clone()
        clone_label = clone_label.detach().cpu()
        clone_inputs = inputs.clone()
        clone_inputs = clone_inputs.detach().cpu()
        clone_output = output.clone()
        clone_output = clone_output.detach().cpu()
        # [batch_size, input_length, feature_size]
        print("done")
        print(idx.size(0))
        for j in range(30):
            label_sum = 0
            output_sum = 0
            input_sum = 0
            for k in range(idx.size(0)):
                label_sum += clone_label[k, -1, j]
                output_sum += clone_output[k, -1, j]
                input_sum += clone_inputs[k, -1, j]
            label_list.append(label_sum / (idx.size(0)))
            output_list.append(output_sum / (idx.size(0)))
            input_list.append(input_sum / (idx.size(0)))
        # else:
        #    loss = criterion(output[-output_window:], label[-output_window:])
        total_loss += loss.item()
        prev_uni_losses += prev_loss
        test_uni_losses += uni_loss


    plt.plot(label_list, color="r", label="acc")
    plt.plot(output_list, color=(0,0,0),label="pre")
    plt.plot(input_list, color="b", label="prev")
    plt.legend()
    plt.show()
    prev_uni_loss = prev_uni_losses / len(data_source)
    test_uni_loss = test_uni_losses / len(data_source)

    print("test loss: ", end='')
    print(total_loss / len(data_source))
    return test_uni_loss, prev_uni_loss


# In[80]:
print('preprocess_data')
result, pick = get_preprocess_result()
matrics = result[1]

train_dataset = DatasetPrice(matrics[0])
valid_dataset = DatasetPrice(matrics[1])
test_dataset = DatasetPrice(matrics[2])
print(len(train_dataset))
print(len(valid_dataset))
print(len(test_dataset))

print("loading data")

train_data_loader = DataLoaderPrice(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
valid_data_loader = DataLoaderPrice(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_data_loader = DataLoaderPrice(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


# train_data, val_data, scaler = get_data()
model = TransAm().to(device)
print(sum(p.numel() for p in model.parameters()))
print(sum(p.numel() for p in model.parameters() if p.requires_grad))
criterion = nn.MSELoss()
lr = 0.005
# optimizer = torch.optim.SGD(model.parameters(), lr=lr)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.98)

best_val_loss = 1000000
epochs = 8  # The number of epochs
best_model = None

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(train_dataset)
    '''
    if (epoch % 1 is 0):
        val_loss = plot(model, valid_dataset, epoch, scaler)
        # predict_future(model, val_data,200,epoch,scaler)
    else:
    '''
    val_loss = evaluate(model, valid_dataset)

    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.5f} | valid ppl {:8.2f}'.format(epoch,
                                                                                                  (time.time() - epoch_start_time),
                                                                                                  val_loss,
                                                                                                  math.exp(val_loss)))
    print('-' * 89)

    if val_loss < best_val_loss:
        print("update best model")
        best_val_loss = val_loss
        best_model = model

    scheduler.step()


test_uni_loss, prev_uni_loss = test(model, test_dataset)
gain = 1 - (test_uni_loss/prev_uni_loss)
print("final model")
print(gain)
print(test_uni_loss)
print(prev_uni_loss)

test_uni_loss, prev_uni_loss = test(best_model, test_dataset)
gain = 1 - (test_uni_loss/prev_uni_loss)
print("best model")
print(gain)
print(test_uni_loss)
print(prev_uni_loss)
'''
for name,param in model.named_parameters():
    print(name, param.shape, param.mean())
'''


