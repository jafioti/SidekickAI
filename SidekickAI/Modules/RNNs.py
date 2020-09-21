from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
from enum import Enum

import torch.nn.functional as F
import csv, random, re, os, math, time
from SidekickAI.Modules.Attention import ContentAttention, LuongAttn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CONTAINS ALL RNN BASED SEQUENCE MODELS (RNN ENCODERS, RNN DECODERS, SEQ2SEQ, etc.)

# A basic wrapper around the RNN module to allow for stacked biRNN modules
class BiRNNModule(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type, n_layers, dropout=0.):
        super().__init__()
        assert (rnn_type == nn.RNN or rnn_type == nn.LSTM or rnn_type == nn.GRU), "rnn_type must be a valid RNN type (torch.RNN, torch.LSTM, or torch.GRU)"
        self.n_layers = n_layers
        self.forward_rnn = rnn_type(input_size, hidden_size, num_layers=n_layers, dropout=0 if n_layers == 1 else dropout)
        self.backward_rnn = rnn_type(input_size, hidden_size, num_layers=n_layers, dropout=0 if n_layers == 1 else dropout)

    def forward(self, x):
        #X: (seq len, batch size, features) or (num directions, seq len, batch size, features)
        if len(x.shape) == 4 and x.shape[0] == 1: x.unsqueeze_(0)
        if len(x.shape) == 4:
            forward_out, forward_hidden = self.forward_rnn(x[0])
            backward_out, backward_hidden = self.backward_rnn(x[1])
        else:
            forward_out, forward_hidden = self.forward_rnn(x)
            backward_out, backward_hidden = self.backward_rnn(torch.flip(x, dims=[0]))
        return torch.stack((forward_out, backward_out), dim=0), torch.stack((forward_hidden, backward_hidden), dim=0)

# RNN-Based bidirectional encoder that takes entire sequence at once and returns output sequence along with final hidden state
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, rnn_type, dropout=0.):
        super().__init__()
        assert (rnn_type == nn.RNN or rnn_type == nn.LSTM or rnn_type == nn.GRU), "rnn_type must be a valid RNN type (torch.RNN, torch.LSTM, or torch.GRU)"
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type

        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size, num_layers=n_layers, dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, inputs, lengths=None, hidden=None): # Takes the entire input sequence at once
        # inputs: (seq_len, batch_size, embed_dim)
        batch_size = inputs.shape[1]
        seq_len = inputs.shape[0]
        
        # Pack if lengths are provided
        if lengths is not None: inputs = nn.utils.rnn.pack_padded_sequence(inputs, lengths, enforce_sorted=False)
        # Push through RNN layer
        outputs, hidden = self.rnn(inputs) # the hidden state defaults to zero when not provided
        # Unpack if lengths are provided
        if lengths is not None: outputs, _ =  nn.utils.rnn.pad_packed_sequence(outputs)

        # Select hidden state if using LSTM
        if self.rnn_type == nn.LSTM: hidden = hidden[0]
        # Concat bidirectional hidden states
        hidden = hidden.view(self.n_layers, 2, batch_size, self.hidden_size)
        hidden = torch.cat((hidden[:, 0], hidden[:, 1]), dim=-1)
        # Concat bidirectional outputs
        outputs = outputs.view(seq_len, batch_size, 2, self.hidden_size)
        outputs = torch.cat((outputs[:, :, 0], outputs[:, :, 1]), dim=-1)

        return outputs, hidden

class PyramidRNNEncoder(nn.Module): # MAY NEED TO FIX SHAPES
    def __init__(self, input_size, n_layers, rnn_type, dropout=0.):
        super().__init__()
        assert (rnn_type == nn.RNN or rnn_type == nn.LSTM or rnn_type == nn.GRU), "rnn_type must be a valid RNN type (torch.RNN, torch.LSTM, or torch.GRU)"
        self.n_layers = n_layers
        self.dropout = dropout
        self.rnn_type = rnn_type
        self.final_hidden_size = int(input_size * math.pow(2., n_layers + 1))

        self.rnns = nn.ModuleList([BiRNNModule(input_size=int(input_size * math.pow(2., float(i))), hidden_size=int(input_size * math.pow(2., float(i))), n_layers=1, rnn_type=rnn_type) for i in range(1, n_layers + 1)])

    def forward(self, inputs, lengths=None, hidden=None):
        # inputs: (seq_len, batch_size, embed_dim)
        # Shave off end to ensure it will fit
        inputs = inputs[:- int(inputs.shape[0] % math.pow(2, self.n_layers))]
        seq_len, batch_size, features = inputs.shape
        #assert seq_len % math.pow(2, self.n_layers) == 0, "Sequence Length must be divisible by " + str(int(math.pow(2, self.n_layers))) + " to work with the Pyramid RNN Encoder"
        
        # Pack if lengths are provided
        if lengths is not None: inputs = nn.utils.rnn.pack_padded_sequence(inputs, lengths, enforce_sorted=False)

        # Feed through rnn layers
        for i in range(len(self.rnns)):
            # Concat neighbors
            inputs = inputs.contiguous().view(2, int(inputs.shape[1] / 2), batch_size, int(inputs.shape[-1] * 2)) if i > 0 else inputs.contiguous().view(int(inputs.shape[0] / 2), batch_size, int(inputs.shape[-1] * 2))
            # Feed through layer
            inputs, hidden = self.rnns[i](inputs)
            # Dropout
            if i != self.n_layers - 1: F.dropout(inputs, self.dropout)

        # Concat final hiddens
        hidden = hidden.view(1, 2, batch_size, self.final_hidden_size)
        hidden = torch.cat((hidden[:, 0], hidden[:, 1]), dim=-1)[-1]
        # Concat final outputs
        inputs = inputs.view(inputs.shape[0], batch_size, 2, self.final_hidden_size)
        inputs = torch.cat((inputs[:, :, 0], inputs[:, :, 1]), dim=-1)

        return inputs, hidden

# Decoder based RNN using Luong Attention
class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, dropout=0.1):
        super().__init__()

        # Keep for reference
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.gru = nn.GRU(input_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = ContentAttention(hidden_size, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # Note: we run this one step (word) at a time
        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(input_step, last_hidden)
        # Calculate context vector from the current GRU output
        context = self.attn(rnn_output.squeeze(0), encoder_outputs.transpose(0, 1), return_weighted_sum=True)
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        concat_input = torch.cat((rnn_output.squeeze(0), context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word
        output = F.softmax(self.out(concat_output), dim=1)

        return output, hidden

class Seq2SeqRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_vocab, encoder_layers, decoder_layers, input_vocab=None, dropout=0., teacher_forcing_ratio=1., max_length=200):
        super().__init__()
        self.hyperparameters = locals()
        self.output_embedding = nn.Embedding(output_vocab.num_words, hidden_size)
        if input_vocab is not None: self.input_embedding = nn.Embedding(input_vocab.num_words, hidden_size) if input_vocab != output_vocab else self.output_embedding # If input and output vocabs are the same, reuse embeddings
        self.encoder = EncoderRNN(input_size=hidden_size, hidden_size=hidden_size, n_layers=encoder_layers, rnn_type=nn.GRU, dropout=dropout)
        self.decoder = LuongAttnDecoderRNN(hidden_size, hidden_size * 2, output_vocab.num_words, decoder_layers, dropout)
        self.input_vocab, self.output_vocab = input_vocab, output_vocab
        self.max_length = max_length
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.convert_input = nn.Linear(input_size, hidden_size) if input_size != hidden_size else None
    
    def forward(self, input_seq, target_seq=None):
        # Ensure there is no SOS_token at the start of the input seq or target seq
        if self.input_vocab is not None and input_seq[0, 0].item() == self.input_vocab.SOS_token: input_seq = input_seq[1:]
        if target_seq is not None and target_seq[0, 0].item() == self.output_vocab.SOS_token: target_seq = target_seq[1:]
        # Warn if there is no EOS token at the end of the target
        if target_seq is not None and not (target_seq[-1] == self.output_vocab.EOS_token).any(): print("Warning: There is no EOS token at the end of the target passed to the model!")

        input_lengths = torch.LongTensor([len(input_seq[:, i][(input_seq[:, i] != self.input_vocab.PAD_token)]) for i in range(input_seq.shape[1])]).to(device) if self.input_vocab is not None else torch.LongTensor([len(input_seq) for i in range(input_seq.shape[1])]).to(device)
        if self.input_vocab is not None: input_seq = self.input_embedding(input_seq)
        if self.convert_input is not None: input_seq = self.convert_input(input_seq)
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_lengths)
        # Create initial decoder input (start with SOS tokens for each sentence)
        decoder_input = torch.LongTensor([[self.output_vocab.SOS_token for _ in range(input_seq.shape[1])]])
        decoder_input = decoder_input.to(device)

        # Set initial decoder hidden state to the encoder's final hidden state
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]

        # Forward batch of sequences through decoder one time step at a time
        final_outputs = torch.empty((target_seq.shape[0], target_seq.shape[1], self.output_vocab.num_words), device=device) if target_seq is not None else torch.empty((self.max_length, input_seq.shape[1], self.output_vocab.num_words), device=device)
        for t in range(target_seq.shape[0] if target_seq is not None else self.max_length):
            decoder_output, decoder_hidden = self.decoder(self.output_embedding(decoder_input), decoder_hidden, encoder_outputs)
            final_outputs[t] = decoder_output
            # Teacher forcing / Autoregressive
            decoder_input = target_seq[t].view(1, -1) if random.random() < self.teacher_forcing_ratio and target_seq is not None else torch.argmax(decoder_output, dim=-1).view(1, -1)
            if target_seq is None and torch.argmax(decoder_output, dim=-1)[0].item() == self.output_vocab.EOS_token: return final_outputs[:t+1]

        return final_outputs