import torch
import torch.nn as nn
from enum import Enum

import torch.nn.functional as F
import unittest, pprint
from SidekickAI.Modules.RNNs import Seq2SeqRNN, EncoderRNN, DecoderRNN, PyramidEncoderRNN, BiRNNModule
from SidekickAI.Modules.Transformers import TransformerAggregator, TransformerEncoder, Seq2SeqTransformer
from SidekickAI.Modules.Attention import ContentAttention
from SidekickAI.Data import vocab

class RNNTests(unittest.TestCase):
    '''Test RNN Modules'''
    def test_RNN_Encoder(self):
        model = EncoderRNN(input_size=10, hidden_size=20, n_layers=3, rnn_type=nn.GRU, dropout=0.1, bidirectional=True)
        inp = torch.rand(20, 5, 10) # Seq len = 20, batch size = 5, dim = 10
        out, hidden = model(inp)
        self.assertTrue(out.shape[0] == 20 and out.shape[1] == 5 and out.shape[2] == 40, "Output is not correct shape")
        self.assertTrue(hidden.shape[0] == 3 and hidden.shape[1] == 5 and hidden.shape[2] == 40, "Output hidden is not correct shape")
    
    def test_RNN_Decoder(self):
        model = DecoderRNN(input_size=10, hidden_size=20, output_size=45, n_layers=3, rnn_type=nn.GRU, dropout=0.1)
        inp = torch.rand(5, 10) # batch size = 5, input dim = 10
        encoder_output = torch.rand(10, 5, 20) # Seq len = 10, batch size = 5, dim = 20
        hidden = torch.rand(3, 5, 20) # n_layers = 3, batch size = 5, hidden dim = 20
        out, hidden = model(inp, hidden, encoder_output)
        self.assertTrue(out.shape[0] == 5 and out.shape[1] == 45, "Output is not correct shape")
        self.assertTrue(hidden.shape[0] == 3 and hidden.shape[1] == 5 and hidden.shape[2] == 20, "Output hidden is not correct shape")

    def test_RNN_Seq2Seq(self):
        local_vocab = vocab.getAlphabetVocab()
        model = Seq2SeqRNN(input_size=10, hidden_size=20, target_vocab=local_vocab, encoder_layers=2, decoder_layers=2, input_vocab=local_vocab, dropout=0.1, device=torch.device("cpu"), max_length=50)
        inp = torch.full((15, 5), fill_value=local_vocab.num_words - 1, dtype=torch.long) # seq len = 15, batch size = 5, input dim = 10
        out = model(inp)
        self.assertTrue(out.shape[0] <= 50 and out.shape[1] == 5 and out.shape[2] == local_vocab.num_words, "Autoregressive output is not correct shape")
        target = torch.full((25, 5), fill_value=local_vocab.num_words - 1, dtype=torch.long)
        target[-1] = torch.full(tuple([5]), fill_value=local_vocab.EOS_token, dtype=torch.long)
        out = model(inp, target) # Use fake target sequence to run with teacher forcing
        self.assertTrue(out.shape[0] == 25 and out.shape[1] == 5 and out.shape[2] == local_vocab.num_words, "Teacher-forced output is not correct shape")
        
class TransformerTests(unittest.TestCase):
    '''Test Transformer Modules'''
    def test_Transformer_Encoder(self):
        local_vocab = vocab.getAlphabetVocab()
        model = TransformerEncoder(input_size=10, hidden_size=20, num_heads=2, num_layers=3, forward_expansion=2, input_vocab=local_vocab, dropout=0.1, max_len=50, learned_pos_embeddings=False, device=torch.device("cpu"))
        inp = torch.full((15, 5), fill_value=local_vocab.num_words - 1, dtype=torch.long, device=torch.device("cpu")) # seq len = 15, batch size = 5
        out = model(inp)
        self.assertTrue(out.shape[0] == 15 and out.shape[1] == 5 and out.shape[2] == 20, "Output is not correct shape")

    def test_Transformer_Aggregator(self):
        local_vocab = vocab.getAlphabetVocab()
        model = TransformerAggregator(input_size=10, hidden_size=20, output_size=40, num_heads=2, num_layers=3, forward_expansion=2, input_vocab=local_vocab, dropout=0.1, max_len=50, learned_pos_embeddings=False, device=torch.device("cpu"))
        inp = torch.full((15, 5), fill_value=local_vocab.num_words - 1, dtype=torch.long, device=torch.device("cpu")) # seq len = 15, batch size = 5
        out = model(inp)
        self.assertTrue(out.shape[0] == 5 and out.shape[1] == 40, "Output is not correct shape")

    def test_Transformer_Seq2Seq(self):
        local_vocab = vocab.getAlphabetVocab()
        model = Seq2SeqTransformer(input_size=10, hidden_size=20, target_vocab=local_vocab, num_heads=2, num_encoder_layers=2, num_decoder_layers=2, forward_expansion=2, input_vocab=local_vocab, dropout=0.1, max_len=50, learned_pos_embeddings=True, device=torch.device("cpu"))
        inp = torch.full((15, 5), fill_value=local_vocab.num_words - 1, dtype=torch.long) # seq len = 15, batch size = 5
        out = model(inp)
        self.assertTrue(out.shape[0] <= 50 and out.shape[1] == 5 and out.shape[2] == local_vocab.num_words, "Autoregressive output is not correct shape")
        target = torch.full((25, 5), fill_value=local_vocab.num_words - 1, dtype=torch.long)
        target[-1] = torch.full(tuple([5]), fill_value=local_vocab.EOS_token, dtype=torch.long)
        out = model(inp, target) # Use fake target sequence to run with teacher forcing
        self.assertTrue(out.shape[0] == 25 and out.shape[1] == 5 and out.shape[2] == local_vocab.num_words, "Teacher-forced output is not correct shape")

class AttentionTests(unittest.TestCase):
    '''Test Attention Modules'''
    def test_Content_Attention(self):
        model = ContentAttention(query_hidden_size=50, key_hidden_size=40)
        queries = torch.rand(5, 50)
        keys = torch.rand(5, 10, 40)
        out = model(queries, keys, return_weighted_sum=True)
        self.assertTrue(out.shape[0] == 5 and out.shape[1] == 40, "Output with weighted avg is not correct shape")
        out = model(queries, keys, return_weighted_sum=False)
        self.assertTrue(out.shape[0] == 5 and out.shape[1] == 10, "Output without weighted avg is not correct shape")

def test(): 
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(unittest.makeSuite(RNNTests))
    runner.run(unittest.makeSuite(TransformerTests))
    runner.run(unittest.makeSuite(AttentionTests))

if __name__ == "__main__": test()