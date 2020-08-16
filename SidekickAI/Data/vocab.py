# Sidekick Vocab v1.0
# This file contains the Vocab class and teh function to load the BertWordPeice vocab

class Vocab:
    def __init__(self, name, PAD_token=0, SOS_token=1, EOS_token=2):
        self.name = name
        self.trimmed = False
        self.word2index = {"PAD": PAD_token, "SOS": SOS_token, "EOS": EOS_token}
        self.word2count = {"PAD": 0, "SOS":0, "EOS":0}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  # Count SOS, EOS, PAD
        self.PAD_token = PAD_token # Used for padding short sentences
        self.SOS_token = SOS_token # Start-of-sentence token
        self.EOS_token = EOS_token # End-of-sentence token

    def addSentence(self, sentence, custom_tokenization_function=None): # Add sentence tokenized by spaces or custom function
        words = sentence.split(' ') if custom_tokenization_function is None else custom_tokenization_function(sentence)
        for word in words:
            self.addWord(word)

    def addWord(self, word): # Add single token to vocab
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1
    
    def addList(self, ls): # Add list of tokens to vocab
        for i in range(len(ls)):
            self.addWord(ls[i])

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

    def fit_counts_to_corpus(self, corpus, tokenizer=None): # Count the token frequency in the corpus and normalize
        if tokenizer is None:
            from SidekickAI.Data import tokenization
            tokenizer = tokenization.tokenize
        # Count
        def count_tokens(corpus):
            if isinstance(corpus, list):
                for i in range(len(corpus)):
                    count_tokens(corpus[i])
            else:
                tokens = tokenizer(corpus)
                for i in range(len(tokens)):
                    self.word2count[tokens[i]] += 1
        count_tokens(corpus)
        # Normalize
        max_count = max(list(self.word2count.values()))
        for i in range(len(list(self.word2count.values()))):
            self.word2count[list(self.word2count.keys())[i]] /= max_count

    # Recursivly gets indexes
    def indexes_from_tokens(self, tokens):
        if tokens is None:
            return
        if len(tokens) == 0:
            return
        if isinstance(tokens[0], list):
            current = []
            for i in range(len(tokens)):
                nextLevel = self.indexes_from_tokens(tokens[i])
                if nextLevel is not None:
                    current.append(nextLevel)
            return current
        elif isinstance(tokens, list):
            return [self.word2index[word] for word in tokens]
        else:
            return self.word2index[tokens]

    # Recursivly gets tokens
    def tokens_from_indexes(self, indexes):
        if indexes is None:
            return []
        if isinstance(indexes, int):
            return self.index2word[int(indexes)]
        if len(indexes) == 0:
            return []
        if isinstance(indexes[0], list):
            current = []
            for i in range(len(indexes)):
                nextLevel = self.tokens_from_indexes(indexes[i])
                if nextLevel is not None:
                    current.append(nextLevel)
            return current
        elif isinstance(indexes, list):
            return [self.index2word[int(index)] for index in indexes]

    def contains_token(self, token):
        return token in self.word2index

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
    vocabCounts = open(os.path.join(current_dir, "wordCounts.txt"), "r", encoding="utf-8").readlines()
    vocabCounts = [float(i.replace('\n', '')) for i in vocabCounts]
    for i in range(len(voc.word2count)):
        voc.word2count[list(voc.word2count.keys())[i]] = vocabCounts[i]
    return(voc)

# Make a vocab containing the alphabet and puncuation
def getAlphabetVocab(additional_tokens=None):
    assert additional_tokens is None or (isinstance(additional_tokens, list) and isinstance(additional_tokens[0], str))
    from string import ascii_lowercase
    voc = Vocab("alphabetVocab")
    for letter in ascii_lowercase:
        voc.addWord(str(letter))
    voc.addList([".", "'", "!", "?", " ", ",", ";", "-"])
    if additional_tokens is not None:
        for i in range(len(additional_tokens)):
            voc.addWord(additional_tokens[i])
    return voc