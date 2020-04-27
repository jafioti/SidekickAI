# Sidekick Vocab v1.0
# This file contains the Vocab class and teh function to load the BertWordPeice vocab

class Vocab:
    def __init__(self, name, PAD_token=0, SOS_token=1, EOS_token=2):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  # Count SOS, EOS, PAD
        self.PAD_token = PAD_token # Used for padding short sentences
        self.SOS_token = SOS_token # Start-of-sentence token
        self.EOS_token = EOS_token # End-of-sentence token

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {self.PAD_token: "PAD", self.SOS_token: "SOS", self.EOS_token: "EOS"}
        self.num_words = 3 # Count default tokens

        for word in keep_words:
            self.addWord(word)

#Creates a vocab for the bert wordpeice tokenizer
def getBertWordPieceVocab(additional_tokens=None):
    assert additional_tokens is None or (isinstance(additional_tokens, list) and isinstance(additional_tokens[0], str))
    voc = Vocab("bertVocab")
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    vocabLines = open(os.path.join(current_dir, "bert-base-uncased-vocab.txt"), "rb").readlines()
    for i in range(len(vocabLines)):
        voc.addWord(vocabLines[i].decode("utf-8").replace("\n", "").strip())
    if additional_tokens is not None:
        for i in range(len(additional_tokens)):
            voc.addWord(additional_tokens[i])
    return(voc)