from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
from enum import Enum

import torch.nn.functional as F
import csv, random, re, os, math
from SidekickAI.Models.Attention import MultiHeadAttention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EmbeddingTransformerSeq2Seq(nn.Module):
    def __init__(self, input_size, input_vocab, target_vocab, num_heads, num_encoder_layers, num_decoder_layers, forward_expansion, dropout=0.1, max_len=50, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__()
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

    def create_pad_mask(self, seq, pad_idx):
        # seq shape: (seq len, batch size)
        mask = seq.transpose(0, 1) == pad_idx
        # mask shape: (batch size, seq len) <- PyTorch transformer wants this shape for mask
        return mask

    def forward(self, src, trg=None):
        src_len, batch_size, dim = src.shape
        trg_len = trg.shape[0] if trg is not None else 1
        if trg is None:
            autoregressive = True
            trg = torch.full((1, batch_size), fill_value=self.target_vocab.SOS_token, dtype=torch.float).to(self.device)
            final_out = torch.zeros((self.max_len, batch_size, self.target_vocab.num_words)).to(self.device) # To hold the distributions
        else:
            autoregressive = False
            assert trg[0][0].item() == self.target_vocab.SOS_token # Ensure there is an SOS token at the start of the trg

        for i in range(self.max_len if autoregressive else 1):
            # Get pad masks
            src_pad_mask = self.create_pad_mask(src, self.input_vocab.PAD_token)
            trg_pad_mask = self.create_pad_mask(trg, self.target_vocab.PAD_token)

            # Make position tensors
            src_positions = torch.arange(0, src_len).unsqueeze(1).expand(src_len, batch_size).to(self.device)
            trg_positions = torch.arange(0, trg_len).unsqueeze(1).expand(trg_len, batch_size).to(self.device)

            # Add position embeddings to input embeddings
            src = self.dropout(src + self.src_positional_embedding(src_positions))
            trg = self.dropout(trg + self.trg_positional_embedding(trg_positions))

            # Get target subsequent mask
            trg_subsequent_mask = self.transformer.generate_square_subsequent_mask(trg_len).to(self.device)

            # Training, just a single forward pass is needed
            out = self.transformer(src=src, tgt=trg, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=trg_pad_mask, tgt_mask=trg_subsequent_mask)
            out = self.fc_out(out)
            # out shape: (trg_len, batch size, target_num_words)

            if autoregressive:
                final_out[i] = out[-1]
                trg = torch.cat((trg, torch.argmax(out[-1], dim=-1)), dim=0)
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


        # out shape: (trg_len, batch size, dim)
        return out