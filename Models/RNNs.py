from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
from enum import Enum

import torch.nn.functional as F
import csv, random, re, os, math
from Sidekick.Models.Attention import ContentAttention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CONTAINS ALL RNN BASED SEQUENCE MODELS (RNN ENCODERS, RNN DECODERS, SEQ2SEQ, etc.)

# RNN-Based bidirectional encoder that takes entire sequence at once and returns output sequence along with final hidden state
class RNNEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, rnn_type, dropout=0):
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
        hidden = torch.cat((hidden[:, 0], hidden[:, 1]), dim=-1)[-1]
        # Concat bidirectional outputs
        outputs = outputs.view(seq_len, batch_size, 2, self.hidden_size)
        outputs = torch.cat((outputs[:, :, 0], outputs[:, :, 1]), dim=-1)

        return outputs, hidden

# RNN-Based autoregressive decoder that takes in a single timestep at once
class RNNDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers, rnn_type, dropout=0):
        super().__init__()
        assert (rnn_type == nn.RNN or rnn_type == nn.LSTM or rnn_type == nn.GRU), "rnn_type must be a valid RNN type (torch.RNN, torch.LSTM, or torch.GRU)"

        self.input = nn.Linear(input_size, hidden_size)
        self.rnn = rnn_type(hidden_size, hidden_size, num_layers=1, dropout=dropout)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        #X: (1, batch_size, embedding_size)
        # Scale up input to hidden size
        x = self.input(x)
        #Pass through GRU
        outputs, hidden = self.rnn(x, hidden)
        # Scale up to output size
        outputs = self.out(outputs)
        return(outputs, hidden)

# RNN-Based Encoder-Decoder Seq2Seq with optional attention, fixed decoding steps, and dynamic stopping
class RNNSeq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, encoder_layers, decoder_layers, local_vocab, max_target_length, encoder_rnn_type=nn.GRU, decoder_rnn_type=nn.GRU, use_attention=False, dynamic_stopping=False, custom_embedding=nn.Embedding, dropout=0):
        super().__init__()
        # Layers
        if custom_embedding == nn.Embedding: # Allow other embeddings to optionally be loaded, or set custom_embedding to -1 to not embed at all
            self.embedding = nn.Embedding(num_embeddings=local_vocab.num_words, embedding_dim=input_size)
        else:
            self.embedding = custom_embedding
        self.encoder = RNNEncoder(input_size=input_size, hidden_size=hidden_size, n_layers=encoder_layers, rnn_type=encoder_rnn_type, dropout=dropout)
        self.decoder = RNNDecoder(input_size=input_size, hidden_size=hidden_size * 2 if use_attention else hidden_size, output_size=local_vocab.num_words, n_layers=decoder_layers, rnn_type=decoder_rnn_type, dropout=dropout)
        self.encoder_bi_to_uni = nn.Linear(hidden_size * 2, hidden_size)
        if use_attention: 
            self.attention_query = nn.Linear(hidden_size, hidden_size)
            self.attention = ContentAttention(hidden_size=hidden_size)
        if dynamic_stopping: self.dynamic_stopper = nn.Sequential(nn.Linear(hidden_size, math.floor(hidden_size / 2)), nn.ReLU(), nn.Linear(math.floor(hidden_size / 2), 1), nn.Sigmoid()) # Feed forward net for determining the 'stopping probability' given the current hidden state

        # Values
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.local_vocab = local_vocab
        self.max_target_length = max_target_length
        self.use_attention = use_attention
        self.dynamic_stopping = dynamic_stopping

    def forward(self, input_seq, input_lengths, decoding_steps=None, return_embeddings=False):
        assert not return_embeddings or self.embedding is not None, "In order to return embeddings, self.embedding must be an embedding layer!"
        # Input Seq: (seq length, batch size) or (seq length, batch size, embedding dim)
        batch_size = input_seq.shape[1]
        seq_len = input_seq.shape[0]
        # Embed
        if self.embedding is not None:
            input_seq = self.embedding(input_seq)

        # Run through encoder
        encoded_input, encoder_hidden = self.encoder(input_seq, input_lengths)

        # Convert encoded_input and encoder_hidden from hidden_size * 2 to hidden_size
        if self.use_attention: encoded_input = self.encoder_bi_to_uni(encoded_input)
        decoder_hidden = self.encoder_bi_to_uni(encoder_hidden)

        # Setup decoder variables
        # Starting input for the decoder
        decoder_input = self.embedding(torch.full([batch_size], self.local_vocab.SOS_token, dtype=torch.long).to(device))
        # The final output tensor holding either vocab distributions or embeddings
        final_output = torch.full([(decoding_steps if decoding_steps is not None else self.max_target_length), batch_size] + ([self.local_vocab.num_words] if not return_embeddings else []), fill_value=self.local_vocab.PAD_token if return_embeddings else 0).to(device)
        final_output = self.embedding(final_output) if return_embeddings else final_output.scatter(dim=-1, index=torch.LongTensor([self.local_vocab.PAD_token]), value=1)
        # The output lengths
        final_lengths = torch.LongTensor(batch_size).zero_().to(device)
        # The dynamic stopping prob and controller index (only used if we are dynamically stopping)
        if self.dynamic_stopping and decoding_steps is None: 
            dynamic_stop_prob = torch.zeros(batch_size).to(device)
        idx = torch.BoolTensor(batch_size).fill_(1).to(device)

        # Decoding loop
        for decoding_step in range(decoding_steps if decoding_steps is not None else self.max_target_length):
            # Get context vector if nessacary            
            if self.use_attention:
                context_vector[idx] = self.attention(query=self.attention_query(decoder_hidden[idx]), values=encoded_input[:, idx])
                decoder_input[idx] = torch.cat(decoder_input[idx], context_vector[idx], dim=-1)
            # Feed through decoder
            output_logits[idx], decoder_hidden[idx] = self.decoder(decoder_input[idx], decoder_hidden[idx])

            # Convert output logits to final output distributions / embeddings
            output_distribution[idx] = F.softmax(output_logits[idx], dim=-1)
            decoder_input[idx] = torch.matmul(output_distribution[idx], self.embedding.weight) if return_embeddings else self.embedding(torch.argmax(output_distribution[idx], dim=-1))

            # Get dynamic stopping probability and dynamically stop if applicable
            if self.dynamic_stopping and decoding_steps is None: 
                dynamic_stop_prob[idx] += self.dynamic_stopper(decoder_hidden[idx]) # Get new stop prob
                idx = (dynamic_stop_prob < 1 - 0.01).data # Update selector
                final_lengths[idx] = decoding_step + 1 # Update output lengths
                if not idx.any(): return final_output[:decoding_step + 1], 1 - dynamic_stop_prob, final_lengths # Stop when we are done and return outputs, inverse stop probs (for ponder costs), and output lengths

            # Insert output into final output tensor
            final_output[decoding_step, idx] = (decoder_input[idx] if return_embeddings else output_distribution[idx]) * (1 - dynamic_stopping_prob[idx] if self.dynamic_stopping and decoding_steps is None else 1)

            # Check if EOS token was outputted if we are not dynamically stopping
            if not self.dynamic_stopping or decoding_steps is not None:
                idx = (torch.argmax(output_distribution, dim=-1) != self.local_vocab.EOS_token and idx).data
                if not idx.any(): return final_output[:decoding_step + 1], final_lengths # Stop when we are done and return outputs and output lengths

        return final_output, final_lengths