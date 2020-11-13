from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
from enum import Enum

import torch.nn.functional as F
from torch.distributions import Categorical
import csv, random, re, os, math
from SidekickAI.Modules.Attention import MultiHeadAttention
from SidekickAI.Utilities.functional import weighted_avg, batch_dot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Seq2SeqTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, target_vocab, num_heads, num_encoder_layers, num_decoder_layers, forward_expansion, input_vocab=None, dropout=0.1, max_len=50, learned_pos_embeddings=False, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        '''A Seq2Seq Transformer which embeds inputs and outputs distributions over the output vocab\n
        Init Inputs:
            input_size (int): The size of embeddings in the network
            hidden_size (int): The size of hidden vectors in the network
            target_vocab (vocab): The target vocab
            num_heads (int): The number of heads in both the encoder and decoder
            num_encoder_layers (int): The number of layers in the transformer encoder
            num_decoder_layers (int): The number of layers in the transformer decoder
            forward_expansion (int): The factor of expansion in the elementwise feedforward layer
            input_vocab (vocab) [Default: None]: The input vocab, if none, then inputs are already expected as vectors
            dropout (float) [Default: 0.1]: The amount of dropout
            max_len (int) [Default: 50]: The max target length used when a target is not provided
            learned_pos_embeddings (bool) [Default: False]: To use learned positional embeddings or fixed ones
            device (torch.device): The device that the network will run on
        Inputs:
            src (Tensor): The input sequence of shape (src length, batch size)
            trg (Tensor) [default=None]: The target sequence of shape (trg length, batch size)
        Returns:
            output (Tensor): The return sequence of shape (target length, batch size, target tokens)'''
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.hyperparameters = locals()
        self.device = device
        self.input_vocab = input_vocab
        self.target_vocab = target_vocab
        self.max_len = max_len
        self.input_size = input_size
        if input_vocab is not None: self.src_embedding = nn.Embedding(input_vocab.num_words, input_size)
        self.src_positional_embedding = nn.Embedding(max_len, input_size) if learned_pos_embeddings else self.generate_pos_embeddings
        self.trg_embedding = nn.Embedding(target_vocab.num_words, input_size)
        self.trg_positional_embedding = nn.Embedding(max_len, input_size) if learned_pos_embeddings else self.generate_pos_embeddings
        self.transformer = nn.Transformer(d_model=hidden_size, dim_feedforward=hidden_size * forward_expansion, nhead=num_heads, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_size, target_vocab.num_words)
        self.convert_input = nn.Linear(input_size, hidden_size) if input_size != hidden_size else None

    def generate_pos_embeddings(self, seq): # Generate statc positional embeddings of shape (seq len, batch size, embed size)
        '''seq: (seq len, 1) or (seq len)'''
        pe = torch.zeros(seq.shape[0], self.input_size, device=self.device)
        position = torch.arange(0, seq.shape[0], dtype=torch.float, device=self.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.input_size, 2).float().to(device) * (-math.log(10000.0) / self.input_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        return pe[:, :]

    def forward(self, src, trg=None):
        '''src: (seq len, batch size)\n
        trg: (seq len, batch size) or None'''
        if len(src.shape) == 2: src_len, batch_size = src.shape
        else: src_len, batch_size, input_dims = src.shape
        assert src_len < self.max_len, "Input was too large! The input must be less than " + str(self.max_len) + " tokens!"

        # Handle target given/autoregressive
        if trg is None:
            autoregressive = True
            trg = torch.full((1, batch_size), fill_value=self.target_vocab.SOS_token, dtype=torch.long, device=self.device)
        else:
            autoregressive = False
            if trg[0][0].item() != self.target_vocab.SOS_token: # Ensure there is an SOS token at the start of the trg, add if there isn't
                trg = torch.cat((torch.full((1, batch_size), fill_value=self.target_vocab.SOS_token, dtype=torch.long, device=self.device), trg), dim=0)
            if any(trg[-1, :] == self.target_vocab.EOS_token): # Ensure there is no EOS token in the target
                # Make lists without EOS tokens
                temp_trg = [row[(row != self.target_vocab.EOS_token)] for row in trg.transpose(0, 1)]
                trg = torch.stack(temp_trg).transpose(0, 1)
        
        # Embed src
        src_positions = torch.arange(0, src_len, device=self.device).unsqueeze(1).expand(src_len, batch_size)
        if len(src.shape) == 2:
            src_pad_mask = (src == self.input_vocab.PAD_token).transpose(0, 1)
            src_embed = self.src_embedding(src) + self.src_positional_embedding(src_positions)
        else:
            src_pad_mask = None
            src_embed = src + self.src_positional_embedding(src_positions)
        # Convert src input_dims to hidden_dims if nessacary
        if self.convert_input is not None: src_embed = self.convert_input(src_embed)

        for i in range(self.max_len if autoregressive else 1):
            # Get target pad mask
            trg_pad_mask = (trg == self.target_vocab.PAD_token).transpose(0, 1)

            # Embed target
            trg_positions = torch.arange(0, trg.shape[0], device=self.device).unsqueeze(1).expand(trg.shape[0], batch_size)
            trg_embed = self.trg_embedding(trg) + self.trg_positional_embedding(trg_positions)
            if self.convert_input is not None: trg_embed = self.convert_input(trg_embed)

            # Get target subsequent mask
            trg_subsequent_mask = self.transformer.generate_square_subsequent_mask(trg.shape[0]).to(self.device)

            # Feed through model
            out = self.transformer(src=src_embed, tgt=trg_embed, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=trg_pad_mask, tgt_mask=trg_subsequent_mask)
            out = F.softmax(self.fc_out(out), dim=-1)
            # out shape: (trg_len, batch size, target_num_words)

            if autoregressive:
                # Get the soft embedding
                dist = Categorical(out[-1])
                trg = torch.cat((trg, dist.sample().unsqueeze(0)), dim=0)
                if all([any(trg[:, x] == self.target_vocab.EOS_token) for x in range(batch_size)]): # EOS was outputted in all batches
                    break

        # out shape: (trg_len, batch size, target_num_words)
        return out

class TransformerEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, num_layers, forward_expansion, input_vocab=None, dropout=0.1, max_len=50, learned_pos_embeddings=False, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        '''A Transformer emcoder which encodes inputs into encoded vectors\n
        Init Inputs:
            input_size (int): The size of embeddings in the network
            hidden_size (int): The size of hidden vectors in the network
            num_heads (int): The number of heads in both the encoder and decoder
            num_layers (int): The number of layers in the transformer encoder
            forward_expansion (int): The factor of expansion in the elementwise feedforward layer
            input_vocab (vocab) [Default: None]: The input vocab, if none, then inputs are already expected as vectors
            dropout (float) [Default: 0.1]: The amount of dropout
            max_len (int) [Default: 50]: The max target length used when a target is not provided
            learned_pos_embeddings (bool) [Default: False]: To use learned positional embeddings or fixed ones
            device (torch.device): The device that the network will run on
        Inputs:
            src (Tensor): The input sequence of shape (src length, batch size)
        Returns:
            output (Tensor): The return sequence of shape (target length, batch size, target tokens)'''
        super().__init__()
        self.hyperparameters = locals()
        self.device = device
        self.input_vocab = input_vocab
        self.max_len = max_len
        self.input_size = input_size
        if input_vocab is not None: self.src_embedding = nn.Embedding(input_vocab.num_words, input_size)
        self.src_positional_embedding = nn.Embedding(max_len, input_size) if learned_pos_embeddings else self.generate_pos_embeddings
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size * forward_expansion, dropout=dropout), num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.convert_input = nn.Linear(input_size, hidden_size) if input_size != hidden_size else None

    def generate_pos_embeddings(self, seq): # Generate statc positional embeddings of shape (seq len, batch size, embed size)
        '''seq: (seq len, 1) or (seq len)'''
        pe = torch.zeros(seq.shape[0], self.input_size, device=self.device)
        position = torch.arange(0, seq.shape[0], dtype=torch.float, device=self.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.input_size, 2).float().to(self.device) * (-math.log(10000.0) / self.input_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        return pe[:, :]

    def forward(self, src):
        '''src: (seq len, batch size) or (seq len, batch size, embed dims)'''
        if self.input_vocab is not None and len(src.shape) == 2: src_len, batch_size = src.shape
        else: src_len, batch_size, input_dims = src.shape
        assert src_len < self.max_len, "Input was too large! The input must be less than " + str(self.max_len) + " tokens!"

        # Embed src
        src_positions = torch.arange(0, src_len, device=self.device).unsqueeze(1).expand(src_len, batch_size)
        if self.input_vocab is not None and len(src.shape) == 2:
            src_pad_mask = (src == self.input_vocab.PAD_token).transpose(0, 1)
            src_embed = self.src_embedding(src) + self.src_positional_embedding(src_positions)
        else:
            src_pad_mask = None
            src_embed = src + self.src_positional_embedding(src_positions)
        # Convert src input_dims to hidden_dims if nessacary
        if self.convert_input is not None: src_embed = self.convert_input(src_embed)
        out = self.transformer_encoder(src=src_embed, src_key_padding_mask=(src == self.input_vocab.PAD_token).transpose(0, 1) if self.input_vocab is not None and len(src.shape) == 2 else None)
        return out

class TransformerAggregator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_heads, num_layers, forward_expansion, input_vocab=None, dropout=0.1, max_len=50, learned_pos_embeddings=False, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        '''A Transformer aggregator which encodes inputs into a single aggregated vector\n
        Init Inputs:
            input_size (int): The size of embeddings in the network
            hidden_size (int): The size of hidden vectors in the network
            output_size (int): The size of the aggregated vector
            num_heads (int): The number of heads in both the encoder and decoder
            num_layers (int): The number of layers in the transformer encoder
            forward_expansion (int): The factor of expansion in the elementwise feedforward layer
            input_vocab (vocab) [Default: None]: The input vocab, if none, then inputs are already expected as vectors
            dropout (float) [Default: 0.1]: The amount of dropout
            max_len (int) [Default: 50]: The max target length used when a target is not provided
            learned_pos_embeddings (bool) [Default: False]: To use learned positional embeddings or fixed ones
            device (torch.device): The device that the network will run on
        Inputs:
            src (Tensor): The input sequence of shape (src length, batch size) or (src length, batch size, embed size)
        Returns:
            output (Tensor): The return sequence of shape (target length, batch size, target tokens)'''
        super().__init__()
        self.encoder = TransformerEncoder(input_size=input_size, hidden_size=hidden_size, num_heads=num_heads, num_layers=num_layers, forward_expansion=forward_expansion, input_vocab=input_vocab, dropout=dropout, max_len=max_len, learned_pos_embeddings=learned_pos_embeddings, device=device)
        self.out = nn.Linear(hidden_size, output_size)
        self.aggregate_input_vector = nn.Parameter(torch.randn(input_size, requires_grad=True), requires_grad=True) # Learned input vector for the aggregating position, like CLS for BERT

    def forward(self, input_seq):
        '''src (Tensor): The input sequence of shape (src length, batch size) or (src length, batch size, embed size)\n
        The input sequence should NOT have an aggregating vector (CLS) already on it.'''
        # Append aggregate vector to the beginning
        input_seq = torch.cat((self.aggregate_input_vector.unsqueeze(0).unsqueeze(0).repeat(1, input_seq.shape[1], 1), (self.encoder.src_embedding(input_seq) if len(input_seq.shape) == 2 else input_seq)), dim=0)
        output = self.encoder(input_seq)
        return self.out(output[0])