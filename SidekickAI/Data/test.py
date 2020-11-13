import unittest

from SidekickAI.Data import tokenization, vocab, batching

class TokenizationTests(unittest.TestCase):
    '''Test Tokenization'''
    def test_Wordpiece_Tokenization(self):
        tokens = tokenization.tokenize_wordpiece("Hello, how are you?")
        self.assertTrue(tokens == ['hello', ',', 'how', 'are', 'you', '?'])
        tokens = tokenization.tokenize_wordpiece(["Hello, how are you", "I'm doing well, thank you.", ["What's up man", "not much"]])
        self.assertTrue(tokens == [['hello', ',', 'how', 'are', 'you'], ['i', "'", 'm', 'doing', 'well', ',', 'thank', 'you', '.'], [['what', "'", 's', 'up', 'man'], ['not', 'much']]])

    def test_Alphabet_Tokenization(self):
        tokens = tokenization.tokenize_alphabet("Hello, how are you?")
        self.assertTrue(tokens == ['h', 'e', 'l', 'l', 'o', ',', ' ', 'h', 'o', 'w', ' ', 'a', 'r', 'e', ' ', 'y', 'o', 'u', '?'])
        tokens = tokenization.tokenize_alphabet(["Hello, how are you", "I'm doing well, thank you.", ["What's up man", "not much"]])
        self.assertTrue(tokens == [['h', 'e', 'l', 'l', 'o', ',', ' ', 'h', 'o', 'w', ' ', 'a', 'r', 'e', ' ', 'y', 'o', 'u'], ['i', "'", 'm', ' ', 'd', 'o', 'i', 'n', 'g', ' ', 'w', 'e', 'l', 'l', ',', ' ', 't', 'h', 'a', 'n', 'k', ' ', 'y', 'o', 'u', '.'], [['w', 'h', 'a', 't', "'", 's', ' ', 'u', 'p', ' ', 'm', 'a', 'n'], ['n', 'o', 't', ' ', 'm', 'u', 'c', 'h']]])

    def test_Moses_Tokenization(self):
        tokens = tokenization.tokenize_moses("Hello, how are you?")
        self.assertTrue(tokens == ['hello', ',', 'how', 'are', 'you', '?'])
        tokens = tokenization.tokenize_moses(["Hello, how are you", "I'm doing well, thank you.", ["What's up man", "not much"]])
        self.assertTrue(tokens == [['hello', ',', 'how', 'are', 'you'], ['i', '&apos;m', 'doing', 'well', ',', 'thank', 'you', '.'], [['what', '&apos;s', 'up', 'man'], ['not', 'much']]])
    
    def test_Spacy_Tokenization(self):
        tokens = tokenization.tokenize_spacy("Hello, how are you?")
        self.assertTrue(tokens == ['Hello', ',', 'how', 'are', 'you', '?'])
        tokens = tokenization.tokenize_spacy(["Hello, how are you", "I'm doing well, thank you.", ["What's up man", "not much"]])
        self.assertTrue(tokens == [['Hello', ',', 'how', 'are', 'you'], ['I', "'m", 'doing', 'well', ',', 'thank', 'you', '.'], [['What', "'s", 'up', 'man'], ['not', 'much']]])

def test():
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(unittest.makeSuite(TokenizationTests))

if __name__ == "__main__": test()