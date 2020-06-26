from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv, random, re, os
import unicodedata
import codecs
from io import open
import itertools
import math
import prepareData

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Encoder
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, embedding, n_layers=1, dropout=0):
        super(Encoder, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=n_layers, dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        self.hiddenConverter1 = nn.Linear(hidden_size * 2, hidden_size * 2) # Layer to convert two bidirectional hidden states to one unidirectional hidden state
        self.hiddenConverter2 = nn.Linear(hidden_size * 2, hidden_size) # Layer to convert two bidirectional hidden states to one unidirectional hidden state

    def forward(self, inputs, hidden=None): # Takes the entire input sequence at once
        # inputs.shape = (batch_size, seq_len, embed_dim)
        batch_size = inputs.shape[0]
        # Initialize hidden state
        hidden = self.init_hidden(batch_size)
        # Push through RNN layer (the ouput is irrelevant)
        _, hidden = self.gru(inputs.transpose(0, 1)) # the hidden state defaults to zero when not provided
        hidden = self._flatten_hidden(hidden, batch_size)
        # Scale down to single direction hidden state
        hidden = self.hiddenConverter1(hidden)
        hidden = F.relu(hidden)
        hidden = self.hiddenConverter2(hidden)
        return hidden

    def _flatten_hidden(self, h, batch_size):
        if isinstance(h, tuple): # LSTM
            X = torch.cat([self._flatten(h[0], batch_size), self._flatten(h[1], batch_size)], 1)
        else: # GRU
            X = self._flatten(h, batch_size)
        return X

    def _flatten(self, h, batch_size): # Takes a tensor and flattens it to (batch size, -1)
        # (num_layers*num_directions, batch_size, hidden_dim)  ==>
        # (batch_size, num_directions*num_layers, hidden_dim)  ==>
        # (batch_size, num_directions*num_layers*hidden_dim)
        return h.transpose(0,1).contiguous().view(batch_size, -1)

    def init_hidden(self, batch_size):
        return torch.zeros(self.n_layers * 2, batch_size, self.hidden_size).to(device)

# Decoder
class Decoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, output_size, n_layers=1, dropout=0):
        super(Decoder, self).__init__()
        self.input = nn.Linear(embedding_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=1, dropout=dropout)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        #X: (1, batch_size, embedding_size)
        # Scale up input to hidden size
        x = self.input(x)
        #Pass through GRU
        outputs, hidden = self.gru(x, hidden)
        # Scale up to output distribution
        outputs = self.out(outputs)
        return(outputs, hidden)

# Latent Transformation
class LatentTransformation(nn.Module):
    def __init__(self, input_size, layers, output_size, activation=F.relu):
        super(LatentTransformation, self).__init__()
        # Create layers
        self.layers = nn.ModuleList([nn.Linear(input_size, layers[0])])
        self.layers.extend([nn.Linear(layers[i - 1], layers[i]) for i in range(1, len(layers))])
        self.layers.append(nn.Linear(layers[-1], output_size))
        # Initialize weights
        for i in range(len(self.layers)):
            torch.nn.init.xavier_uniform_(self.layers[i].weight)
        self.activation = activation

    def forward(self, x):
        #X: (batch_size, input_size)
        length = len(self.layers)
        for i in range(length):
            if i < length - 1:
                x = self.activation(self.layers[i](x))
            if i == length - 1:
                x = self.layers[i](x)
        #X: (batch_size, output_size)
        return(x)

# Main model
class ConstrainedSeq2Seq(nn.Module):
    def __init__(self, hidden_size, vocab_size, embedding_size=100, encoder_layers=1, latent_layers=1, decoder_layers=1, training_enc_dec=True, embedding=None, dropout=0, device=device):
        super(ConstrainedSeq2Seq, self).__init__()
        self.device = device
        if embedding == None:
            self.embedding = nn.Embedding(vocab_size, embedding_size).to(device)
        else:
             self.embedding = embedding.to(device)
        self.encoder = Encoder(input_size=embedding_size, hidden_size=hidden_size, embedding=self.embedding, n_layers=encoder_layers, dropout=dropout).to(device)
        self.latent_transformation = LatentTransformation(input_size=hidden_size, layers=[hidden_size for i in range(latent_layers)], output_size=hidden_size).to(device)
        self.decoder = Decoder(embedding_size=embedding_size, hidden_size=hidden_size, output_size=vocab_size, n_layers=decoder_layers, dropout=dropout).to(device)
        self.training_enc_dec = training_enc_dec
        if training_enc_dec:
            self.setTrainingEncDec()
        else:
            self.setTrainingLatentTransformation()

    def setTrainingEncDec(self):
        self.training_enc_dec = True
        # Unfreeze enc and dec
        for param in self.encoder.parameters():
                param.requires_grad = True
        for param in self.decoder.parameters():
                param.requires_grad = True
        for param in self.embedding.parameters():
                param.requires_grad = True
        # Freeze latent transformation (not technically needed)
        for param in self.latent_transformation.parameters():
            param.requires_grad = False

    def setTrainingLatentTransformation(self):
        self.training_enc_dec = False
        # Freeze enc and dec
        for param in self.encoder.parameters():
                param.requires_grad = False
        for param in self.decoder.parameters():
                param.requires_grad = False
        for param in self.embedding.parameters():
                param.requires_grad = False
        # Unfreeze latent transformation
        for param in self.latent_transformation.parameters():
            param.requires_grad = True

    def forward(self, x, target=None, maxOutputLength=20):
        #X: (seq_len, batch_size, 1)
        #target: (seq_len, batch_size, 1)
        batch_size = x.shape[1]
        x = self.embedding(x)
        #X: (seq_len, batch_size, hidden_size)
        x.transpose_(0, 1)
        #X: (batch_size, seq_len, hidden_size)

        # Run through encoder
        encoder_hidden = self.encoder(x)
        encoder_hidden.unsqueeze_(0)

        #encoded: (seq len, batch_size, hidden_size)
        #encoder_hidden: (1, batch_size, hidden_size) 4 because of the combination of hidden states from back and front RNNS

        # Run through latent transformation
        if not self.training_enc_dec:
            encoder_hidden = self.latent_transformation(encoder_hidden)
        #encoded: (seq_len, batch_size, hidden_size)

        # Run through decoder
        final_outputs = []
        decoder_input = torch.LongTensor([1 for i in range(batch_size)]).to(self.device).unsqueeze(0) # make starting tokens for all members of batch: (1, batch_size, 1)
        decoder_hidden = encoder_hidden
        if target is not None:
            runLength = target.shape[0]
        else:
            runLength = maxOutputLength
        for i in range(runLength):
            # embed input
            decoder_input = self.embedding(decoder_input).to(device)
            #decoder_input: (1, batch_size, embedding size)

            # run through rnn
            decoded_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            decoded_output = F.log_softmax(decoded_output, dim=2)
            #decoded_output: (1, batch_size, vocab_size)

            # handle new input
            if target is not None:
                # Use teacher forcing
                decoder_input = target[i].unsqueeze(0)
            else:
                _, topi = decoded_output.topk(1)
                decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]]).to(self.device)
            # handle current output
            if self.training:
                final_outputs.append(decoded_output.squeeze(0))
            else:
                final_outputs.append(decoder_input.squeeze(0))
                # if we are evaluating and the batch size is 1, then quit when eos is generated
                if final_outputs[-1][0] == prepareData.EOS_token:
                    break

        return(final_outputs)
