from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
from enum import Enum

import torch.nn.functional as F
from torch.distributions import Categorical
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
        if len(x.shape) == 4 and x.shape[0] == 1: x.squeeze_(0)
        # Pack
        lengths = torch.IntTensor([x.shape[-3] for i in range(x.shape[-2])])
        if len(x.data.shape) == 4:
            forward_out, forward_hidden = self.forward_rnn(nn.utils.rnn.pack_padded_sequence(x[0], lengths, enforce_sorted=False))
            backward_out, backward_hidden = self.backward_rnn(nn.utils.rnn.pack_padded_sequence(x[1], lengths, enforce_sorted=False))
        else:
            forward_out, forward_hidden = self.forward_rnn(nn.utils.rnn.pack_padded_sequence(x, lengths, enforce_sorted=False))
            backward_out, backward_hidden = self.backward_rnn(nn.utils.rnn.pack_padded_sequence(torch.flip(x, dims=[0]), lengths, enforce_sorted=False))
        # Unpack
        forward_out, _ = nn.utils.rnn.pad_packed_sequence(forward_out)
        #forward_hidden, _ = nn.utils.rnn.pad_packed_sequence(forward_hidden)
        backward_out, _ = nn.utils.rnn.pad_packed_sequence(backward_out)
        #backward_hidden, _ = nn.utils.rnn.pad_packed_sequence(backward_hidden)
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
        
        # Pack and use lengths if provided
        if lengths is not None: 
            inputs = nn.utils.rnn.pack_padded_sequence(inputs, lengths, enforce_sorted=False)
        else:
            inputs = nn.utils.rnn.pack_padded_sequence(inputs, torch.LongTensor([seq_len for i in range(batch_size)]), enforce_sorted=False)
        # Push through RNN layer
        outputs, hidden = self.rnn(inputs) # the hidden state defaults to zero when not provided
        # Unpack
        outputs, _ =  nn.utils.rnn.pad_packed_sequence(outputs)

        # Select hidden state if using LSTM
        if self.rnn_type == nn.LSTM: hidden = hidden[0]
        # Concat bidirectional hidden states
        hidden = hidden.view(self.n_layers, 2, batch_size, self.hidden_size)
        hidden = torch.cat((hidden[:, 0], hidden[:, 1]), dim=-1)
        # Concat bidirectional outputs
        outputs = outputs.view(seq_len, batch_size, 2, self.hidden_size)
        outputs = torch.cat((outputs[:, :, 0], outputs[:, :, 1]), dim=-1)

        return outputs, hidden

class PyramidEncoderRNN(nn.Module): # MAY NEED TO FIX SHAPES
    def __init__(self, input_size, n_layers, rnn_type, dropout=0.):
        super().__init__()
        assert (rnn_type == nn.RNN or rnn_type == nn.LSTM or rnn_type == nn.GRU), "rnn_type must be a valid RNN type (torch.RNN, torch.LSTM, or torch.GRU)"
        self.input_size = input_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.rnn_type = rnn_type
        self.pad_vector = torch.nn.Parameter(torch.randn((1, 1, input_size), requires_grad=True))

        self.rnns = nn.ModuleList([BiRNNModule(input_size=int(input_size * math.pow(2., float(i))), hidden_size=int(input_size * math.pow(2., float(i))), n_layers=1, rnn_type=rnn_type) for i in range(1, n_layers + 1)])

    def forward(self, input_seq, lengths=None, hidden=None):
        '''inputs: \n
            input_seq: (seq_len, batch_size, input_dim)
            lengths: (batch size)
            hidden: 
        outputs:
            seq: (seq_len / 2 ^ n_layers + 1, batch size, input_dim * 2 ^ n_layers + 1)
            hidden: None'''
        seq_len, batch_size, features = input_seq.shape
        
        # Pack if lengths are provided
        if lengths is not None: input_seq = nn.utils.rnn.pack_padded_sequence(input_seq, lengths, enforce_sorted=False)

        # Pad to ensure input_seq.shape[0] is divisible by 2^(n_layers - 1)
        if input_seq.shape[0] % math.pow(2, self.n_layers) != 0: input_seq = torch.cat((input_seq, self.pad_vector.repeat(int(math.pow(2, self.n_layers) - (input_seq.shape[0] % math.pow(2, self.n_layers))), batch_size, 1)), dim=0)

        # Feed through rnn layers
        for i in range(len(self.rnns)):
            # Concat neighbors
            input_seq = input_seq.transpose(1, 2) if i > 0 else input_seq.transpose(0, 1) # Reshape messes up if batch dim does not come first
            input_seq = input_seq.contiguous().reshape(2, batch_size, int(input_seq.shape[2] / 2), int(input_seq.shape[3] * 2)) if i > 0 else input_seq.contiguous().reshape(batch_size, int(input_seq.shape[1] / 2), int(input_seq.shape[2] * 2))
            input_seq = input_seq.transpose(1, 2) if i > 0 else input_seq.transpose(0, 1)
            # Feed through layer
            input_seq, hidden = self.rnns[i](input_seq)
            # Dropout
            if i != self.n_layers - 1: F.dropout(input_seq, self.dropout)
        # Concat final hiddens
        hidden = hidden.transpose(0, 1)
        hidden = torch.cat((hidden[:, 0], hidden[:, 1]), dim=-1)[-1]
        # Concat final outputs
        input_seq = torch.cat((input_seq[0], input_seq[1]), dim=-1)

        # return input_seq, hidden
        return input_seq, None

# Decoder based RNN using Luong Attention
class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, use_attention=True, n_layers=1, dropout=0.1):
        super().__init__()

        # Keep for reference
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.use_attention = use_attention

        # Define layers
        self.gru = nn.GRU(input_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        if use_attention: 
            self.concat = nn.Linear(hidden_size * 2, hidden_size)
            self.attn = ContentAttention(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # Note: we run this one step at a time
        # input_step: (1, batch size, hidden dim)
        # Pack
        input_step = nn.utils.rnn.pack_padded_sequence(input_step, torch.LongTensor([1 for i in range(input_step.shape[1])]), enforce_sorted=False)
        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(input_step, last_hidden)
        rnn_output, _ = nn.utils.rnn.pad_packed_sequence(rnn_output)
        output = rnn_output.squeeze(0)

        if self.use_attention:
            # Calculate context vector from the current GRU output
            context = self.attn(rnn_output.squeeze(0), encoder_outputs.transpose(0, 1), return_weighted_sum=True)
            # Concatenate weighted context vector and GRU output using Luong eq. 5
            concat_input = torch.cat((rnn_output.squeeze(0), context), 1)
            output = torch.tanh(self.concat(concat_input))
        # Predict next word
        output = F.softmax(self.out(output), dim=-1)

        return output, hidden

class Seq2SeqRNN(nn.Module):
    '''A Seq2Seq RNN which embeds inputs and outputs distributions over the output vocab\n
    Init Inputs:
        inpur_size (int): The size of embeddings / inputs to the network
        hidden_size (int): The RNN hidden size
        target_vocab (vocab): The target vocab
        encoder_layers (int): The number of layers in the transformer encoder
        decoder_layers (int): The number of layers in the transformer decoder
        input_vocab (vocab) [Default: None]: The input vocab, if none, then inputs are already expected as vectors
        dropout (float) [Default: 0.1]: The amount of dropout
        teacher_forcing_ratio (float): The percentage of the time to use teacher forcing
        max_len (int) [Default: 200]: The max target length used when a target is not provided
        device (torch.device): The device that the network will run on
    Inputs:
        src (Tensor): The input sequence of shape (src length, batch size)
        trg (Tensor) [default=None]: The target sequence of shape (trg length, batch size)
    Returns:
        output (Tensor): The return sequence of shape (target length, batch size, target tokens)'''
    def __init__(self, input_size, hidden_size, target_vocab, encoder_layers, decoder_layers, input_vocab=None, use_attention=True, dropout=0., teacher_forcing_ratio=1., max_length=200, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__()
        self.hyperparameters = locals()
        self.device = device
        self.output_embedding = nn.Embedding(target_vocab.num_words, hidden_size)
        if input_vocab is not None: self.input_embedding = nn.Embedding(input_vocab.num_words, hidden_size) if input_vocab != target_vocab else self.output_embedding # If input and output vocabs are the same, reuse embeddings
        self.encoder = EncoderRNN(input_size=hidden_size, hidden_size=hidden_size, n_layers=encoder_layers, rnn_type=nn.GRU, dropout=dropout)
        self.decoder = DecoderRNN(hidden_size, hidden_size * 2, target_vocab.num_words, use_attention, decoder_layers, dropout)
        self.input_vocab, self.target_vocab = input_vocab, target_vocab
        self.max_length = max_length
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.convert_input = nn.Linear(input_size, hidden_size) if input_size != hidden_size else None
    
    def forward(self, input_seq, target_seq=None):
        # Ensure there is no SOS_token at the start of the input seq or target seq
        if self.input_vocab is not None and input_seq[0, 0].item() == self.input_vocab.SOS_token: input_seq = input_seq[1:]
        if target_seq is not None and target_seq[0, 0].item() == self.target_vocab.SOS_token: target_seq = target_seq[1:]
        # Warn if there is no EOS token at the end of the target
        if target_seq is not None and not (target_seq[-1] == self.target_vocab.EOS_token).any(): print("Warning: There is no EOS token at the end of the target passed to the model!")

        input_lengths = torch.IntTensor([len(input_seq[:, i][(input_seq[:, i] != self.input_vocab.PAD_token)]) for i in range(input_seq.shape[1])]) if self.input_vocab is not None else torch.IntTensor([len(input_seq) for i in range(input_seq.shape[1])])
        if self.input_vocab is not None: input_seq = self.input_embedding(input_seq)
        if self.convert_input is not None: input_seq = self.convert_input(input_seq)
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_lengths)
        # Create initial decoder input (start with SOS tokens for each sentence)
        decoder_input = torch.LongTensor([[self.target_vocab.SOS_token for _ in range(input_seq.shape[1])]])
        decoder_input = decoder_input.to(self.device)

        # Set initial decoder hidden state to the encoder's final hidden state
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]

        # Forward batch of sequences through decoder one time step at a time
        final_outputs = torch.empty((target_seq.shape[0], target_seq.shape[1], self.target_vocab.num_words), device=self.device) if target_seq is not None else torch.empty((self.max_length, input_seq.shape[1], self.target_vocab.num_words), device=self.device)
        for t in range(target_seq.shape[0] if target_seq is not None else self.max_length):
            decoder_output, decoder_hidden = self.decoder(self.output_embedding(decoder_input), decoder_hidden, encoder_outputs)
            final_outputs[t] = decoder_output
            # Teacher forcing / Autoregressive
            decoder_input = target_seq[t].view(1, -1) if random.random() < self.teacher_forcing_ratio and target_seq is not None else torch.argmax(decoder_output, dim=-1).view(1, -1)
            if target_seq is None and torch.argmax(decoder_output, dim=-1)[0].item() == self.target_vocab.EOS_token: return final_outputs[:t+1]

        return final_outputs
        
        def forward_beam(self, input_seq, beam_size=1):
            '''Run through model and do beam search decoding\n
            Inputs:
                input_seq (Tensor): The sequence of input tokens of shape (seq len, batch size) or (seq len, batch size, input dim)
                beam_size (int): The size of the beam used during beam search. If beam size is 1, is equivalent to greedy decoding
            Outputs:
                outpsut_seq (list): A list of the output sequence in token indexes for each batch, of shape (seq len, batch size)'''
            # Ensure there is no SOS_token at the start of the input seq or target seq
            if self.input_vocab is not None and input_seq[0, 0].item() == self.input_vocab.SOS_token: input_seq = input_seq[1:]
            if target_seq is not None and target_seq[0, 0].item() == self.target_vocab.SOS_token: target_seq = target_seq[1:]
            # Warn if there is no EOS token at the end of the target
            if target_seq is not None and not (target_seq[-1] == self.target_vocab.EOS_token).any(): print("Warning: There is no EOS token at the end of the target passed to the model!")

            input_lengths = torch.IntTensor([len(input_seq[:, i][(input_seq[:, i] != self.input_vocab.PAD_token)]) for i in range(input_seq.shape[1])]) if self.input_vocab is not None else torch.IntTensor([len(input_seq) for i in range(input_seq.shape[1])])
            if self.input_vocab is not None: input_seq = self.input_embedding(input_seq)
            if self.convert_input is not None: input_seq = self.convert_input(input_seq)
            encoder_outputs, encoder_hidden = self.encoder(input_seq, input_lengths)
            # Create initial decoder input (start with SOS tokens for each sentence)
            decoder_input = torch.LongTensor([[self.target_vocab.SOS_token for _ in range(input_seq.shape[1])]])
            decoder_input = decoder_input.to(self.device)

            # Set initial decoder hidden state to the encoder's final hidden state
            decoder_hidden = encoder_hidden[:self.decoder.n_layers]

            # Run beam search through decoder
            batch_final_outputs = []
            for batch in range(input_seq.shape[1]):
                outputs, final_outputs = [self.target_vocab.SOS_token], []
                best = [[self.target_vocab.SOS_token, 0, decoder_hidden[batch]]] # List for tracking the best tokens, scores, and hidden states
                for t in range(self.max_length):
                    # Feed through every branch and sample outputs
                    sampled_tokens, sampled_scores, sampled_hiddens = [], [], []
                    for i in range(len(best)):
                        if best[i][0] != self.target_vocab.EOS_token:
                            decoder_output, decoder_hidden = self.decoder(self.output_embedding(torch.LongTensor(best[i][0]).unsqueeze(0).unsqueeze(0).to(device)), best[i][2].unsqueeze(0))
                            # Sample beam_size outputs
                            output_dist = Categorical(decoder_output)
                            sampled_tokens.append([output_dist.sample() for i in range(self.beam_width)])
                            sampled_scores.append([output_dist.log_prob(sample) + best[i][1] for sample in sampled_tokens[-1]])
                            best[i][2] = decoder_hidden[0]
                        else:
                            sampled_tokens.append([])
                            sampled_scores.append([])
                    # Select only top beam_width samples
                    top_addresses = [[0, 0]]
                    for i in range(self.beam_width):
                        for branch in range(len(sampled_tokens)):
                            for sample in range(len(sampled_tokens[branch])):
                                if sampled_scores[branch][sample] > sampled_scores[top_addresses[-1][0]][top_addresses[-1][1]] and [branch, sample] not in top_addresses:
                                    top_addresses[-1] = [branch, sample]
                        top_addresses.append([0, 0])
                    best = [[sampled_tokens[i][x], sampled_scores[i][x], best[i][2]] for i, x in top_addresses]
                    outputs = [outputs[i] + sampled_tokens[i][x] for i, x in top_addresses]
                    final_outputs += [[outputs[z], best[z][1]] for z in range(len(best)) if self.target_vocab.EOS_token in outputs[z]]
                batch_final_outputs.append(final_outputs)

        return batch_final_outputs