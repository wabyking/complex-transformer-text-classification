# Model.py

import torch
import torch.nn as nn
from copy import deepcopy
from train_utils import Embeddings,PositionalEncoding
from attention import MultiHeadedAttention
from encoder import EncoderLayer, Encoder
from feed_forward import PositionwiseFeedForward
import numpy as np
from torch.autograd import Variable
from utils import *
import math
from torch.nn.parameter import Parameter

def get_sinusoid_encoding_table(n_src_vocab, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return 1 / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_src_vocab)])

    # sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    # sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)

def get_sinusoid_encoding_table_dim(n_src_vocab, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return 1 / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    # sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_src_vocab)])
    sinusoid_table = np.array([get_posi_angle_vec(1)])
    sinusoid_table=torch.FloatTensor(sinusoid_table)

    enc_output_phase= Parameter(sinusoid_table,requires_grad=True)#虚部向量
    
    sinusoid_table=enc_output_phase.repeat(n_src_vocab,1)

    # sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    # sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    # if padding_idx is not None:
    #     # zero vector for padding dimension
    #     sinusoid_table[padding_idx] = 1.

    return torch.FloatTensor(sinusoid_table)

def get_sinusoid_encoding_table_vocab(n_src_vocab, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''
    def cal_angle(position, hid_idx):
        return 1 / np.power(10000, 2 * (hid_idx // 2) / n_src_vocab)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(n_src_vocab)]

    # sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_src_vocab)])
    sinusoid_table = np.array([get_posi_angle_vec(1)])
    sinusoid_table=torch.FloatTensor(sinusoid_table)

    enc_output_phase= Parameter(sinusoid_table,requires_grad=True)#虚部向量
    
    sinusoid_table=enc_output_phase.repeat(d_hid,1)
    sinusoid_table=sinusoid_table.reshape(n_src_vocab,d_hid)

    # sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    # sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    # if padding_idx is not None:
    #     # zero vector for padding dimension
    #     sinusoid_table[padding_idx] = 1.

    return torch.FloatTensor(sinusoid_table)

class LayerNorm_my(nn.Module):
    "Construct a layer normalization module."
    def __init__(self, d_model, output_size):
        super(LayerNorm_my, self).__init__()
        # w = torch.empty(d_model, output_size).cuda()
        # w=torch.cuda.FloatTensor(w)
        # self.linear_weight = nn.init.normal_(w, mean=0, std=np.sqrt(2.0 / (d_model + output_size)))
        # self.linear_weight = Variable(self.linear_weight, requires_grad=True)
        # self.linear_weight = nn.Parameter(torch.rand(d_model, output_size),requires_grad=True)
        self.linear_weight = Parameter(torch.Tensor(output_size, d_model))
        nn.init.kaiming_uniform_(self.linear_weight, a=math.sqrt(5))

    def forward(self, x,y):

        linear_weight=torch.cuda.FloatTensor(self.linear_weight)
        x=torch.cuda.FloatTensor(x)
        y=torch.cuda.FloatTensor(y)
        x=x.mm(linear_weight.float().t())
        y=y.mm(linear_weight.float().t())
        # y=nn.functional.linear(y,linear_weight.float().t())
        return x,y

class Transformer(nn.Module):
    def __init__(self, config, src_vocab):
        super(Transformer, self).__init__()
        self.config = config
        
        h, N, dropout = self.config.h, self.config.N, self.config.dropout
        d_model, d_ff = self.config.d_model, self.config.d_ff
        self.src_vocab=src_vocab
        
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.encoder_layer=EncoderLayer(config.d_model,deepcopy(attn), deepcopy(ff), dropout)
        self.encoder =Encoder(self.encoder_layer, N)
        
        self.src_word_emb = nn.Embedding(src_vocab, config.d_model, padding_idx=0)
        

        # self.pos_bias = nn.Embedding(src_vocab, config.d_model, padding_idx=0)
        # self.pos_bias = nn.Embedding.from_pretrained(get_sinusoid_encoding_table_dim(src_vocab, config.d_model, padding_idx=0),freeze=True)
        # self.pos_bias = nn.Embedding.from_pretrained(get_sinusoid_encoding_table_vocab(src_vocab, config.d_model, padding_idx=0),freeze=True)
        

        # self.pos_bias = nn.Embedding(1, config.d_model, padding_idx=0)
        # self.pos_bias = nn.Embedding(src_vocab, 1, padding_idx=0)
        # self.position_enc = nn.Embedding(src_vocab, config.d_model, padding_idx=0)
        self.position_enc = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(src_vocab, config.d_model, padding_idx=0),freeze=False)
        

        # position_enc = torch.randn(1000, config.d_model)
        # position_enc = position_enc.unsqueeze(0)
        # self.register_buffer('position_enc', position_enc)

        self.drop = nn.Dropout(p=dropout)
        self.fc = nn.Linear(self.config.d_model,self.config.output_size)
        
        self.softmax = nn.Softmax()

    def forward(self, x):
        # embedded_sents = self.src_embed(x.permute(1,0)) # shape = (batch_size, sen_len, d_model)


        enc_output_real = self.src_word_emb(x.permute(1,0))#实向量
        enc_output_phase= self.position_enc(x.permute(1,0))
        # enc_output_phase= Variable(self.position_enc[:, :enc_output_real.size(1)],requires_grad=True)

        klen=enc_output_phase.size(1)
        pos_seq = torch.arange(klen-1, -1, -1.0, device=enc_output_real.device,dtype=enc_output_real.dtype)
        os_seq=torch.unsqueeze(pos_seq,-1)
        pos_seq=torch.unsqueeze(pos_seq,-1)


        # enc_output_phase=torch.mul(pos_seq,enc_output_phase)+self.pos_bias(x.permute(1,0))
        enc_output_phase=torch.mul(pos_seq,enc_output_phase)


        enc_output = self.drop(enc_output_real)
        enc_output_phase = self.drop(enc_output_phase)

        cos = torch.cos(enc_output_phase)
        sin = torch.sin(enc_output_phase)

        enc_output_real=enc_output*cos
        enc_output_phase=enc_output*sin

        encoded_sents_real,encoded_sents_phase = self.encoder(enc_output_real,enc_output_phase)

        # encoded_sents_real, encoded_sents_phase = self.encoder_layer(enc_output_real, enc_output_phase)
        # encoded_sents_real,encoded_sents_phase = self.encoder(encoded_sents_phase,encoded_sents_real)
        final_feature_map_real = encoded_sents_real[:,-1,:]
        final_feature_map_phase = encoded_sents_phase[:,-1,:]
        # final_out_real,final_out_phase= self.fc(final_feature_map_real,final_feature_map_phase)
        # final_out=final_out_real+final_out_phase
        # encoded_sents=torch.cat([encoded_sents_real,encoded_sents_phase],2)
        # encoded_sents = encoded_sents[:,-1,:] 
        final_feature=final_feature_map_real+final_feature_map_phase
        final_out= self.fc(final_feature)

        return self.softmax(final_out)
    
    def add_optimizer(self, optimizer):
        self.optimizer = optimizer
        
    def add_loss_op(self, loss_op):
        self.loss_op = loss_op
    
    def reduce_lr(self):
        print("Reducing LR")
        for g in self.optimizer.param_groups:
            g['lr'] = g['lr'] / 2
                
    def run_epoch(self, train_iterator, val_iterator, epoch):
        train_losses = []
        val_accuracies = []
        losses = []
        
        # Reduce learning rate as number of epochs increase
        if (epoch == int(self.config.max_epochs/3)) or (epoch == int(2*self.config.max_epochs/3)):
            self.reduce_lr()
            
        for i, batch in enumerate(train_iterator):
            self.optimizer.zero_grad()
            if torch.cuda.is_available():
                x = batch.text.cuda()
                y = (batch.label - 1).type(torch.cuda.LongTensor)
            else:
                x = batch.text
                y = (batch.label - 1).type(torch.LongTensor)
            y_pred = self.__call__(x)
            loss = self.loss_op(y_pred, y)
            loss.backward()
            losses.append(loss.data.cpu().numpy())
            self.optimizer.step()
            self.train()
    
            # if i % 100 == 0:
                # print("Iter: {}".format(i+1))
                # avg_train_loss = np.mean(losses)
                # train_losses.append(avg_train_loss)
                # print("\tAverage training loss: {:.5f}".format(avg_train_loss))
                # losses = []
                
                # Evalute Accuracy on validation set
                # val_accuracy = evaluate_model(self, val_iterator)
                # print("\tVal Accuracy: {:.4f}".format(val_accuracy))
                # train_accuracy = evaluate_model(self, train_iterator)
                # print("\ttrain Accuracy: {:.4f}".format(train_accuracy))
                # print("Iter: {} \tAverage training loss: {:.5f} \tVal Accuracy: {:.4f} \ttrain Accuracy: {:.4f}".format(i+1,avg_train_loss,val_accuracy,train_accuracy))
                
                
        return train_losses, val_accuracies


# Model.py

# import torch
# import torch.nn as nn
# from copy import deepcopy
# from train_utils import Embeddings,PositionalEncoding
# from attention import MultiHeadedAttention
# from encoder import EncoderLayer, Encoder
# from feed_forward import PositionwiseFeedForward
# import numpy as np
# from torch.autograd import Variable
# from utils import *

# def get_sinusoid_encoding_table(n_src_vocab, d_hid, padding_idx=None):
#     ''' Sinusoid position encoding table '''

#     def cal_angle(position, hid_idx):
#         return 1 / np.power(10000, 2 * (hid_idx // 2) / d_hid)

#     def get_posi_angle_vec(position):
#         return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

#     sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_src_vocab)])

#     # sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
#     # sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

#     if padding_idx is not None:
#         # zero vector for padding dimension
#         sinusoid_table[padding_idx] = 0.

#     return torch.FloatTensor(sinusoid_table)

# class Transformer(nn.Module):
#     def __init__(self, config, src_vocab):
#         super(Transformer, self).__init__()
#         self.config = config
        
#         h, N, dropout = self.config.h, self.config.N, self.config.dropout
#         d_model, d_ff = self.config.d_model, self.config.d_ff
        
#         attn = MultiHeadedAttention(h, d_model)
#         ff = PositionwiseFeedForward(d_model, d_ff, dropout)

#         self.encoder_layer=EncoderLayer(config.d_model,deepcopy(attn), deepcopy(ff), dropout)
        
#         self.encoder =Encoder(self.encoder_layer, N)

        
#         self.src_word_emb = nn.Embedding(src_vocab, config.d_model, padding_idx=0)
#         self.src_phase_emb = nn.Embedding(src_vocab, config.d_model, padding_idx=0)

#         # self.position_enc = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(src_vocab, config.d_model, padding_idx=0),freeze=False)
#         # self.position_enc = nn.Embedding(1000, config.d_model, padding_idx=0)
#         # position_enc = torch.randn(src_vocab, config.d_model)
#         # position_enc = position_enc.unsqueeze(0)
#         # self.register_buffer('position_enc', position_enc)

#         # Fully-Connected Layer
#         self.drop = nn.Dropout(p=dropout)
#         self.fc = nn.Linear(
#             self.config.d_model,
#             self.config.output_size
#         )
        
#         # Softmax non-linearity
#         self.softmax = nn.Softmax()

#     def forward(self, x):
#         # embedded_sents = self.src_embed(x.permute(1,0)) # shape = (batch_size, sen_len, d_model)

#         enc_output_real = self.src_word_emb(x.permute(1,0))#实向量

#         # enc_output_phase= Variable(self.position_enc[:, :enc_output_real.size(1)],requires_grad=True)#虚部向量
        
#         enc_output_phase= self.src_phase_emb(x.permute(1,0))#虚部向量

#         klen=enc_output_phase.size(1)

#         pos_seq = torch.arange(klen-1, -1, -1.0, device=enc_output_real.device,dtype=enc_output_real.dtype)
#         os_seq=torch.unsqueeze(pos_seq,-1)
#         pos_seq=torch.unsqueeze(pos_seq,-1)
#         enc_output_phase=torch.mul(pos_seq,enc_output_phase)


#         enc_output = self.drop(enc_output_real)
#         enc_output_phase = self.drop(enc_output_phase)

#         cos = torch.cos(enc_output_phase)+0.0001
#         sin = torch.sin(enc_output_phase)+0.0001

#         enc_output_real=enc_output*cos
#         enc_output_phase=enc_output*sin

#         encoded_sents_real,encoded_sents_phase = self.encoder_layer(enc_output_real,enc_output_phase)
#         encoded_sents_real,encoded_sents_phase = self.encoder(encoded_sents_real,encoded_sents_phase)
        

#         encoded_sents=torch.cat([encoded_sents_real,encoded_sents_phase],2)
#         final_feature_map = encoded_sents[:,-1,:]
#         # final_feature_map_phase = encoded_sents_phase[:,-1,:]
        
#         final_out = self.fc(final_feature_map)
#         # final_out_phase = self.fc(final_feature_map_phase)
#         # final_out=final_out_real+final_out_phase
#         return self.softmax(final_out)
    
#     def add_optimizer(self, optimizer):
#         self.optimizer = optimizer
        
#     def add_loss_op(self, loss_op):
#         self.loss_op = loss_op
    
#     def reduce_lr(self):
#         print("Reducing LR")
#         for g in self.optimizer.param_groups:
#             g['lr'] = g['lr'] / 2
                
#     def run_epoch(self, train_iterator, val_iterator, epoch):
#         train_losses = []
#         val_accuracies = []
#         losses = []
        
#         # Reduce learning rate as number of epochs increase
#         if (epoch == int(self.config.max_epochs/3)) or (epoch == int(2*self.config.max_epochs/3)):
#             self.reduce_lr()
            
#         for i, batch in enumerate(train_iterator):
#             self.optimizer.zero_grad()
#             if torch.cuda.is_available():
#                 x = batch.text.cuda()
#                 y = (batch.label - 1).type(torch.cuda.LongTensor)
#             else:
#                 x = batch.text
#                 y = (batch.label - 1).type(torch.LongTensor)
#             y_pred = self.__call__(x)
#             loss = self.loss_op(y_pred, y)
#             loss.backward()
#             losses.append(loss.data.cpu().numpy())
#             self.optimizer.step()
#             self.train()
    
#             # if i % 100 == 0:
#                 # print("Iter: {}".format(i+1))
#                 # avg_train_loss = np.mean(losses)
#                 # train_losses.append(avg_train_loss)
#                 # print("\tAverage training loss: {:.5f}".format(avg_train_loss))
#                 # losses = []
                
#                 # Evalute Accuracy on validation set
#                 # val_accuracy = evaluate_model(self, val_iterator)
#                 # print("\tVal Accuracy: {:.4f}".format(val_accuracy))
#                 # train_accuracy = evaluate_model(self, train_iterator)
#                 # print("\ttrain Accuracy: {:.4f}".format(train_accuracy))
#                 # print("Iter: {} \tAverage training loss: {:.5f} \tVal Accuracy: {:.4f} \ttrain Accuracy: {:.4f}".format(i+1,avg_train_loss,val_accuracy,train_accuracy))
                
                
#         return train_losses, val_accuracies