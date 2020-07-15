from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
from enum import Enum

import torch.nn.functional as F
import csv, random, re, os, math
from SidekickAI.Modules.Attention import MultiHeadAttention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EmbeddingTransformerSeq2Seq(nn.Module):
    '''A Seq2Seq Transformer which embeds inputs and outputs distributions over the output vocab\n
    Init Inputs:
        input_size (int): The size of embeddings in the network
        input_vocab (vocab): The input vocab
        target_vocab (vocab): The target vocab
        num_heads (int): The number of heads in both the encoder and decoder
        num_encoder_layers (int): The number of layers in the transformer encoder
        num_decoder_layers (int): The number of layers in the transformer decoder
        forward_expansion (int): The factor of expansion in the elementwise feedforward layer
        dropout (float): The amount of dropout
        max_len (int): The max target length used when a target is not provided
        device (torch.device): The device that the network will run on
    Inputs:
        src (Tensor): The input sequence of shape (src length, batch size)
        trg (Tensor) [default=None]: The target sequence of shape (trg length, batch size)
    Returns:
        output (Tensor): The return sequence of shape (target length, batch size, target tokens)'''
    def __init__(self, input_size, input_vocab, target_vocab, num_heads, num_encoder_layers, num_decoder_layers, forward_expansion, dropout=0.1, max_len=50, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__()
        self.hyperparameters = locals()
        self.device = device
        self.input_vocab = input_vocab
        self.target_vocab = target_vocab
        self.max_len = max_len
        self.src_embedding = nn.Embedding(input_vocab.num_words, input_size)
        self.src_positional_embedding = nn.Embedding(max_len, input_size)
        self.trg_embedding = nn.Embedding(target_vocab.num_words, input_size)
        self.trg_positional_embedding = nn.Embedding(max_len, input_size)
        self.transformer = nn.Transformer(d_model=input_size, dim_feedforward=input_size * forward_expansion, nhead=num_heads, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(input_size, target_vocab.num_words)

    def create_pad_mask(self, idx_seq, pad_idx):
        # idx_seq shape: (seq len, batch size)
        mask = idx_seq.transpose(0, 1) == pad_idx
        # mask shape: (batch size, seq len) <- PyTorch transformer wants this shape for mask
        return mask

    def forward(self, src, trg=None):
        src_len, batch_size = src.shape
        trg_len = trg.shape[0] if trg is not None else 1

        # Handle target given/autoregressive
        if trg is None:
            autoregressive = True
            trg = torch.full((1, batch_size), fill_value=self.target_vocab.SOS_token, dtype=torch.long, device=self.device)
            final_out = torch.zeros((self.max_len, batch_size, self.target_vocab.num_words), device=self.device) # To hold the distributions
        else:
            autoregressive = False
            if trg[0][0].item() != self.target_vocab.SOS_token: # Ensure there is an SOS token at the start of the trg, add if there isn't
                trg = torch.cat((torch.full((1, batch_size), fill_value=self.target_vocab.SOS_token, dtype=torch.long, device=self.device), trg), dim=0)
                trg_len += 1
            if trg[-1][0].item() == self.target_vocab.EOS_token: # Ensure there is no EOS token at the end of the trg, remove if there is
                trg = trg[:-1]
                trg_len -= 1
        # Get source pad mask
        src_pad_mask = self.create_pad_mask(src, self.input_vocab.PAD_token)
        # Embed src
        src_positions = torch.arange(0, src_len, device=self.device).unsqueeze(1).expand(src_len, batch_size)
        src_embed = self.dropout(self.src_embedding(src) + self.src_positional_embedding(src_positions))

        for i in range(self.max_len if autoregressive else 1):
            # Get target pad mask
            trg_pad_mask = self.create_pad_mask(trg, self.target_vocab.PAD_token)

            # Embed target
            trg_positions = torch.arange(0, trg_len, device=self.device).unsqueeze(1).expand(trg_len, batch_size)
            trg_embed = self.dropout(self.trg_embedding(trg) + self.trg_positional_embedding(trg_positions))

            # Get target subsequent mask
            trg_subsequent_mask = self.transformer.generate_square_subsequent_mask(trg_len).to(self.device)

            # Training, just a single forward pass is needed
            out = self.transformer(src=src_embed, tgt=trg_embed, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=trg_pad_mask, tgt_mask=trg_subsequent_mask)
            out = self.fc_out(out)
            if not self.training: out = F.softmax(out, dim=-1)
            # out shape: (trg_len, batch size, target_num_words)

            if autoregressive:
                trg_len += 1
                final_out[i] = out[-1]
                trg = torch.cat((trg, torch.argmax(out[-1], dim=-1).unsqueeze(1)), dim=0)
                if all([any(trg[:, x] == self.target_vocab.EOS_token) for x in range(batch_size)]): # EOS was outputted in all batches
                    return final_out[:i + 1]

        # out shape: (trg_len, batch size, target_num_words)
        return out

class VectorTransformerSeq2Seq(nn.Module):
    def __init__(self, input_size, input_vocab_size, target_vocab_size, num_heads, num_encoder_layers, num_decoder_layers, forward_expansion, pad_idx, dropout=0.1, max_len=50, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__()
        self.device = device
        self.pad_idx = pad_idx
        self.max_len = max_len
        self.src_embedding = nn.Embedding(input_vocab_size, input_size)
        self.src_positional_embedding = nn.Embedding(max_len, input_size)
        self.trg_embedding = nn.Embedding(target_vocab_size, input_size)
        self.trg_positional_embedding = nn.Embedding(max_len, input_size)
        self.transformer = nn.Transformer(d_model=input_size, dim_feedforward=input_size * forward_expansion, nhead=num_heads, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def create_pad_mask(self, seq):
        # seq shape: (seq len, batch size)
        mask = seq.transpose(0, 1) == self.pad_idx
        # mask shape: (batch size, seq len) <- PyTorch transformer wants this shape for mask
        return mask

    def forward(self, src, trg):
        src_len, batch_size, dim = src.shape # src_pad_mask: (src_len, batch_size)
        trg_len, batch_size, dim = trg.shape # trg_pad_mask: (trg_len, batch_size)

        # Transpose pad masks (PyTorch transformers want batch size first)
        if src_pad_mask is not None: src_pad_mask.transpose_(0, 1) # src_pad_mask: (batch_size, src_len)
        if trg_pad_mask is not None: trg_pad_mask.transpose_(0, 1) # trg_pad_mask: (batch_size, trg_len)

        # Make position tensors
        src_positions = torch.arange(0, src_len).unsqueeze(1).expand(src_len, batch_size).to(self.device)
        trg_positions = torch.arange(0, trg_len).unsqueeze(1).expand(trg_len, batch_size).to(self.device)

        # Add position embeddings to input embeddings
        src = self.dropout(src + self.src_positional_embedding(src_positions))
        trg = self.dropout(trg + self.trg_positional_embedding(trg_positions))

        # Get target subsequent mask
        trg_subsequent_mask = self.transformer.generate_square_subsequent_mask(trg_len).to(self.device)

        if self.training:
            # Training, just a single forward pass is needed
            out = self.transformer(src=src, tgt=trg, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=trg_pad_mask, tgt_mask=trg_subsequent_mask)
        else:
            # Inference, autoregressive loop through decoder is needed
            for i in range(self.max_len):
                pass


        # out shape: (trg_len, batch size, dim)
        return out