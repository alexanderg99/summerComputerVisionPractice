import copy

import torch
import torchvision
from torch import nn
from torch.nn import functional as F
import numpy as np
import math





def ModelClone(module, N):
    #cloning the layers as required
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

    def forward(self,x):
        return nn.Embedding(self.vocab_size, self.d_model)(x)



class PositionEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        self.d_model = d_model
        self.msl = max_seq_len


        self.pe = torch.zeroes(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0,d_model,2):
                self.pe[pos][i] = math.sin(pos / 10000 ^ (2 * i / self.d_model))
                self.pe[pos][i+1] = math.cos(pos/10000 ^ (2 * (i+1) / self.d_model))

        self.pe = self.pe.unsqueeze(0)
        self.register_buffer('pe', self.pe)

    def forward(self,x):

        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len].cuda()

        return x




class PositionWiseFeedFoward(nn.Module):
    #Doing the FFN Equations

    def __init__(self, d_model, d_ff, dropout=0.1):
        super.__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)


    def forward(self,x):
        return self.w_2(self.dropout(self.w_1(x).relu()))

def _init_weights_(W):
    W = torch.nn.init.normal_(W)
    return

#query: the representation of the word we want to calculate self-attention for
#key: a representatin of each word in the sequence
#value: the real representatio of each word in the sequence.
#Q x K gives us the score that tells us how much weight each value attains in the self-attentionvecotr


def Attention(Q,K,V, d_k=64):

    #query should be the word vector x WQ
    assert type(Q) == type(K) == type(V)
    val = nn.Softmax((Q @ K.transpose(-2,-1) ) / d_k ** 0.5)
    #need to return tensor?
    return val @ V


class MultiHeadAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_k = d_model // h
        self.h = h
        self.d_model = d_model
        #one linear each for Q,K,V and Out

        self.linears = ModelClone(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(p=dropout)

        self.attn = None


    def forward(self,query,key,value):
        #query key value = x,x,x -> the input embedding


        nbatches = query.size(0)
        #linear operation, then split into heads
        query,key,value = [
            lin(x).view(nbatches,-1,self.h, self.d_k).transpose(1,2)
            for lin, x in zip(self.linears,(query,key,value))
        ]


        #attetion part
        x = Attention(query, key, value, self.d_k)

        x = (
            x.transpose(1, 2)
                .contiguous()
                .view(self.nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x)


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers=6):
        super().__init__()
        self.encoderLayers = ModelClone(EncoderBlock(),n_layers)
        self.emb = Embedder(vocab_size, d_model)
        self.pe = PositionEncoder(d_model)
        self.LayerNorm = LayerNormalization(d_model)
        self.n_heads = n_layers

    def forward(self, x, mask):
        x = self.embed(x)
        x = self.pe(x)
        for i in range(self.n_layers):
            x = self.layers[i](x, mask)
        return self.LayerNorm(x)

class Decoder(nn.Module):

    def __init__(self, vocab_size, d_model, n_layers=6):
        super().__init__()
        self.decoderLayers = ModelClone(DecoderBlock(),n_layers)
        self.emb = Embedder(vocab_size, d_model)
        self.pe = PositionEncoder(d_model)
        self.LayerNorm = LayerNormalization(d_model)
        self.n_heads = n_layers

    def forward(self, x, mask):
        x = self.embed(x)
        x = self.pe(x)
        for i in range(self.n_layers):
            x = self.layers[i](x, mask)
        return self.LayerNorm(x)






class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers=6):
        super().__init__()
        self.Encoder = Encoder(vocab_size, d_model, n_layers)
        self.Decoder = Decoder(vocab_size, d_model, n_layers)
        self.Linear = nn.Linear(d_model, vocab_size)
        self.Softmax = nn.Softmax()


    def forward(self, seq, input_mask, output_mask):
        out = self.Encoder(seq, input_mask)
        out = self.Decoder(seq, output_mask)
        out = self.Linear(out)
        out = self.Softmax(out)






#Understanding Layer Normalization

class LayerNormalization(nn.Module):

    def __init__(self, normal_shape, gamma=True, beta=True, epsilon=1e-10):
        super().__init__()

    def forward(self, x):
        u = x.mean(dim=-1, keepdim=True)
        sigma = ((x - u) ** 2).mean(dim=-1, keepdim=True).sqrt()
        result = (x - u)/ sigma
        return result

class Masking(nn.Module):
    #this class zeroes the attention when there are paddings in the iput and output.
    #also this class prevents the decoder from peeking ahead at the output sentence.

    def __init__(self):
        return None


class DecoderBlock():
    def __init__(self, d_model, h, dropout=0.01):
        super.__init__()

        self.LayerNorm1 = LayerNormalization(d_model)
        self.LayerNorm2 = LayerNormalization(d_model)
        self.LayerNorm3 = LayerNormalization(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        self.MultiHeadAttention1 = MultiHeadAttention(h, d_model)
        self.MultiHeadAttention2 = MultiHeadAttention(h, d_model)
        self.FullyConnectedFeedForward = PositionWiseFeedFoward(d_model, d_model)

    def forward(self, x, e_outputs, src_mask, trg_mask):
        x2 = self.LayerNorm1(x)
        x = x + self.dropout_1(self.MultiHeadAttention1(x2, x2, x2, trg_mask))
        x2 = self.LayerNorm2(x)
        x = x + self.dropout_2(self.MultiHeadAttention2(x2, e_outputs, e_outputs,
                                           src_mask))
        x2 = self.LayerNorm3(x)
        x = x + self.dropout_3(self.FullyConnectedFeedForward(x2))
        return x



class EncoderBlock(nn.Module):

    #two sublayers in each block.
    #the output of each sublayer is LayerNorm(x+Sublayer(x))
    #all sublayers in the model produce outputs of dimesnion dmodel = 512

    def __init__(self,d_model, h, dropout=0.01):
        super.__init__()

        self.MultiHeadAttention = MultiHeadAttention(h, d_model)
        self.LayerNorm1 = LayerNormalization(d_model)
        self.LayerNorm2 = LayerNormalization(d_model)
        self.FullyConnectedFeedForward = PositionWiseFeedFoward(d_model, d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)



    def forward(self, x):
        out = self.MultiHeadAttention(self,x,x,x)
        out = self
        out = self.LayerNorm1(x, out)
        out1 = self.FullyConnectedFeedForward(out)
        outfinal = self.LayerNorm2(out1, out)
        return outfinal















