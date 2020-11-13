from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
from enum import Enum

import torch.nn.functional as F
import csv, random, re, os, math

from SidekickAI.Utilities.functional import weighted_avg, batch_dot, batch_matrix_vector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Content based attention (Dot product attention)
class ContentAttention(nn.Module):
    '''Content Based Attention (Dot product attention) over keys using a query'''
    def __init__(self, query_hidden_size, key_hidden_size):
        super().__init__()
        self.query_converter = nn.Linear(query_hidden_size, key_hidden_size) if query_hidden_size != key_hidden_size else None
    
    def forward(self, query, keys, key_mask=None, return_weighted_sum=False):
        '''Inputs:
            query: (batch size, query hidden size)
            keys: (batch size, sequence length, key hidden size)
            key_mask: (batch size, sequence length) [optional]
        Returns:
            distribution = (batch size, sequence length) or average = (batch size, hidden size)'''
        batch_size, sequence_length, key_hidden = keys.shape
        if self.query_converter is not None: query = self.query_converter(query)
        distribution = batch_matrix_vector(x=keys, y=query) # Distribution: (batch size, seq len)
        if key_mask is not None: distribution.data.masked_fill_(key_mask.data, -float('inf'))
        if return_weighted_sum:
            return weighted_avg(keys, F.softmax(distribution, dim=-1))
        else:
            # In training we output log-softmax for NLL, otherwise normal softmax
            return F.log_softmax(distribution, dim=1) if self.training else F.softmax(distribution, dim=1)

class LearnedSeqAttn(nn.Module):
    """Learned attention over a sequence (uses a learned vector to get the attention scores):
    """
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x, x_mask=None, return_avg=False):
        """Input shapes:
            x = (batch size, seq len, hidden size)
            x_mask = (batch size, seq len)
        Returns:
            dist = (batch size, seq len) or avg = (batch size, hidden size)
        """
        x_flat = x.contiguous().view(-1, x.size(-1))
        scores = self.linear(x_flat).view(x.size(0), x.size(1))
        if x_mask is not None: scores.data.masked_fill_(x_mask.data, -float('inf'))
        alpha = F.softmax(scores, dim=1)
        return weighted_avg(x, alpha) if return_avg else alpha

# Multi Head Self Attention from https://github.com/bentrevett/pytorch-seq2seq/blob/master/6%20-%20Attention%20is%20All%20You%20Need.ipynb
class MultiHeadAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, key, value, mask = None):
        '''Inputs:
            query: (batch size, query len, hidden size)
            key: (batch size, key len, hidden size)
            value: (batch size, value len, hidden size)
        Returns:
            x: (batch size, query len, hidden size)'''
        
        batch_size = query.shape[0]
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        #Q = [batch size, query len, hid dim]
        #K = [batch size, key len, hid dim]
        #V = [batch size, value len, hid dim]
                
        # Split into heads
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        #Q = [batch size, n heads, query len, head dim]
        #K = [batch size, n heads, key len, head dim]
        #V = [batch size, n heads, value len, head dim]
                
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        #energy = [batch size, n heads, query len, key len]
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(energy, dim = -1)         
        #attention = [batch size, n heads, query len, key len]
                
        x = torch.matmul(self.dropout(attention), V)
        #x = [batch size, n heads, query len, head dim]
        
        x = x.permute(0, 2, 1, 3).contiguous()
        #x = [batch size, query len, n heads, head dim]
        
        x = x.view(batch_size, -1, self.hid_dim)
        #x = [batch size, query len, hid dim]
        
        x = self.fc_o(x)
        #x = [batch size, query len, hid dim]
        
        return x, attention
        
# Luong attention layer
class LuongAttn(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Get attention scores
        attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)