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