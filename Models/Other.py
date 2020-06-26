from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
from enum import Enum

import torch.nn.functional as F
import csv, random, re, os, math

# Token dropout, drops out tokens with a certian percent probability
class TokenDropout(nn.Module):
    def __init__(self, token_dropout, fill_token, device):
        super().__init__()
        assert token_dropout > 0 and token_dropout < 1, "Token dropout must be between 0 and 1"
        self.token_dropout = token_dropout
        self.fill_token = fill_token
        self.device = device

    def forward(self, inp):
        # inp: (seq len, batch size)
        # Choose probabilities
        probs = torch.empty(inp.shape[0], inp.shape[1]).uniform_(0, 1).to(self.device)
        # Fill in fill_token where the probabilities are less than the dropout
        return torch.where(probs > self.token_dropout, inp, torch.full((inp.shape[0], inp.shape[1]), dtype=torch.long, fill_value=self.fill_token).to(self.device))
