from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
from enum import Enum

import torch.nn.functional as F
import csv, random, re, os, math
from Sidekick.Models.Attention import MultiHeadAttention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformer Implementation from https://github.com/bentrevett/pytorch-seq2seq/blob/master/6%20-%20Attention%20is%20All%20You%20Need.ipynb
class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, n_heads, feedforward_dim, dropout, device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hidden_size)
        self.ff_layer_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(hidden_size, n_heads, dropout, device)
        self.dropout = nn.Dropout(dropout)
        self.positionwise_feedforward = nn.Sequential(nn.Linear(hidden_size, feedforward_dim), nn.ReLU(), self.dropout, nn.Linear(feedforward_dim, hidden_size))
        
    def forward(self, src, src_mask):
        #src = [batch size, src len, hid dim]
        #src_mask = [batch size, src len]
                
        #self attention
        _src, _ = self.self_attention(src, src, src)
        
        #dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        #src = [batch size, src len, hid dim]
        
        #positionwise feedforward
        _src = self.positionwise_feedforward(src)
        
        #dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))
        #src = [batch size, src len, hid dim]
        
        return src

class TransformerEncoder(nn.Module):
    def __init__(self, hidden_dim, n_layers, n_heads, feedforward_dim, dropout, device, max_length = 100):
        super().__init__()

        self.device = device
        self.pos_embedding = nn.Embedding(max_length, hidden_dim)
        self.layers = nn.ModuleList([EncoderLayer(hidden_dim, n_heads, feedforward_dim, dropout, device) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim])).to(device)
        
    def forward(self, src, src_lengths):
        #src = [src len, batch size, hidden dim]
        src.transpose_(0, 1)

        #src = [batch size, src len, hidden dim]
        #src_lengths = [batch size]
        batch_size = src.shape[0]
        src_len = src.shape[1]
        
        # Get src_mask
        src_mask = torch.arange(src_len).to(device)[None, :] < src_lengths[:, None]
        # src_mask = [batch size, src len]

        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        #pos = [batch size, src len]
        
        src = self.dropout((src * self.scale) + self.pos_embedding(pos))
        #src = [batch size, src len, hid dim]
        
        for layer in self.layers:
            src = layer(src, src_mask)
        #src = [batch size, src len, hid dim]
        src.transpose_(0, 1)
        #src = [src len, batch size, hid dim]
            
        return src

class DecoderLayer(nn.Module):
    def __init__(self, hidden_size, n_heads, feedforward_dim, dropout, device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hidden_size)
        self.enc_attn_layer_norm = nn.LayerNorm(hidden_size)
        self.ff_layer_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttentionLayer(hidden_size, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hidden_size, n_heads, dropout, device)
        self.positionwise_feedforward = nn.Sequential(nn.Linear(hidden_size, feedforward_dim), nn.ReLU(), self.dropout, nn.Linear(feedforward_dim, hidden_size))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        
        #trg = [batch size, trg len, hid dim]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, trg len]
        #src_mask = [batch size, src len]
        
        #self attention
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        
        #dropout, residual connection and layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
            
        #trg = [batch size, trg len, hid dim]
            
        #encoder attention
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        
        #dropout, residual connection and layer norm
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))
                    
        #trg = [batch size, trg len, hid dim]
        
        #positionwise feedforward
        _trg = self.positionwise_feedforward(trg)
        
        #dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))
        
        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]
        
        return trg, attention

class TransformerDecoder(nn.Module):
    def __init__(self, output_dim, hidden_size, n_layers, n_heads, feedforward_dim, dropout, device, max_length = 100):
        super().__init__()
        
        self.device = device
        
        self.tok_embedding = nn.Embedding(output_dim, hidden_size)
        self.pos_embedding = nn.Embedding(max_length, hidden_size)
        
        self.layers = nn.ModuleList([DecoderLayer(hidden_size, n_heads, feedforward_dim, dropout, device) for _ in range(n_layers)])
        
        self.fc_out = nn.Linear(hidden_size, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hidden_size])).to(device)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        #trg = [trg len, batch size]
        #enc_src = [src len, batch size, hid dim]
        #trg_mask = [trg len, batch size]
        #src_mask = [src len, batch size]
        trg.transpose_(0, 1)
        enc_src.transpose_(0, 1)
        trg_mask.transpose_(0, 1)
        src_mask.transpose_(0, 1)

        #trg = [batch size, trg len]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, trg len]
        #src_mask = [batch size, src len]
                
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
                            
        #pos = [batch size, trg len]
            
        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
                
        #trg = [batch size, trg len, hid dim]
        
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        
        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]
        
        output = self.fc_out(trg)
        
        #output = [batch size, trg len, output dim]
        
        output.transpose_(0, 1)
        return output, attention

class TransformerSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        
    def make_src_mask(self, src):
        
        #src = [batch size, src len]
        
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        #src_mask = [batch size, 1, 1, src len]

        return src_mask
    
    def make_trg_mask(self, trg):
        
        #trg = [batch size, trg len]
        
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        
        #trg_pad_mask = [batch size, 1, 1, trg len]
        
        trg_len = trg.shape[1]
        
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()
        
        #trg_sub_mask = [trg len, trg len]
            
        trg_mask = trg_pad_mask & trg_sub_mask
        
        #trg_mask = [batch size, 1, trg len, trg len]
        
        return trg_mask

    def forward(self, src, trg):
        #src = [src len, batch size]
        #trg = [trg len, batch size]
        src.transpose_(0, 1)
        trg.transpose_(0, 1)

        #src = [batch size, src len]
        #trg = [batch size, trg len]
                
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        #src_mask = [batch size, 1, 1, src len]
        #trg_mask = [batch size, 1, trg len, trg len]
        
        enc_src = self.encoder(src, src_mask)
        
        #enc_src = [batch size, src len, hid dim]
                
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        
        #output = [batch size, trg len, output dim]
        #attention = [batch size, n heads, trg len, src len]
        
        return output, attention