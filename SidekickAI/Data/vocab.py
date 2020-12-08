# Sidekick Vocab v1.0
# This file contains the Vocab class and teh function to load the BertWordPeice vocab

class Vocab:
    def __init__(self, name, PAD_token=0, SOS_token=1, EOS_token=2, add_default_tokens=True):
        '''The vocab object that contains all data about a single vocabulary.\n
        Vocabularies are simply collections of tokens with mappings to and from indexes, so they can be used for many different things\n
        Inputs:
            name (string): The name of the vocabulary
            PAD_token [Default: 0] (int): The index of the padding token
            SOS_token [Default: 1] (int): The index of the start-of-sentence token
            EOS_token [Default: 2] (int): The index of the end-of-sentence token
            add_default_tokens [Default: true] (bool): Add the PAD, SOS, and EOS tokens automatically
        '''        

        self.name = name
        self.trimmed = False
        if add_default_tokens:
            self.word2index = {"PAD": PAD_token, "SOS": SOS_token, "EOS": EOS_token}
            self.word2count = {"PAD": 0, "SOS":0, "EOS":0}
            self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
            self.num_words = 3  # Count SOS, EOS, PAD
            self.PAD_token = PAD_token # Used for padding short sentences
            self.SOS_token = SOS_token # Start-of-sentence token
            self.EOS_token = EOS_token # End-of-sentence token
        else:
            self.word2index = {}
            self.word2count = {}
            self.index2word = {}
            self.num_words = 0  # Count SOS, EOS, PAD
            self.PAD_token, self.SOS_token, self.EOS_token = None, None, None

    def add_sentence(self, sentence, custom_tokenization_function=None): 
        '''Add sentence tokenized by spaces or custom function\n
        Inputs:
            sentence (string): The sentence with tokens to be added
            custom_tokenization_funtion [Default: None] (function): The function to be used to tokenize the sentence
        '''
        words = sentence.split(' ') if custom_tokenization_function is None else custom_tokenization_function(sentence)
        for word in words:
            self.add_word(word)

    def add_word(self, word):
        '''Add single token to vocab\n
        Inputs:
            word (string): The word to be added as a token to the vocab
        '''
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1
    
    def add_list(self, token_list): 
        '''Add list of tokens to vocab\n
        Inputs:
            token_list (list of strings): A list of tokens to be added to the vocab
        '''
        for i in range(len(token_list)):
            self.add_word(token_list[i])

    def trim(self, min_count):
        '''Remove words from the vocab below a certain count threshold. Requires a count to have been fitted on a corpus of text\n
        Inputs:
            min_count (float): The minimum count a word should have to be kept'''
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {self.PAD_token: "PAD", self.SOS_token: "SOS", self.EOS_token: "EOS"}
        self.num_words = 3 # Count default tokens

        for word in keep_words:
            self.add_word(word)

    def fit_counts_to_corpus(self, corpus, tokenizer=None): 
        '''Count the token frequency in the corpus and normalize over the total number of tokens
        Inputs:
            corpus (string): The corpus of text to be fitted on
            tokenizer [Default: None] (function): The function to be used to tokenize the corpus
        '''
        if tokenizer is None:
            from SidekickAI.Data import tokenization
            tokenizer = tokenization.tokenize_wordpiece
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

    # Recursively gets indexes
    def indexes_from_tokens(self, tokens):
        '''Converts a list or tree of lists of tokens to a list or tree of lists of indexes\n
        Inputs:
            tokens (list or tree of lists of strings): THe collection of tokens to be converted to indexes'''
        if tokens is None:
            print("Is None")
            return
        if len(tokens) == 0:
            print("Zero length")
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
        '''Converts a list or tree of lists of indexes to a list or tree of lists of tokens\n
        Inputs:
            indexes (list or tree of lists of ints): THe collection of indexes to be converted to tokens'''
        if indexes is None:
            return []
        if "Tensor" in str(type(indexes)): # Won't print anything if tensor is passed in
            indexes = indexes.tolist()
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
        '''Check if the vocab contains a certain token\n
        Inputs:
            token (string): The token to check for
        Returns:
            contains (bool): Whether or not the vocab contains the token'''
        return token in self.word2index
    
    def contains_index(self, index):
        '''Check if the vocab contains a certain index\n
        Inputs:
            index (int): The index to check for
        Returns:
            contains (bool): Whether or not the vocab contains the index'''
        return index < len(self.index2word) and index > -1

#Creates a vocab for the bert wordpeice tokenizer
def getBertWordPieceVocab(additional_tokens=None):
    assert additional_tokens is None or (isinstance(additional_tokens, list) and isinstance(additional_tokens[0], str))
    voc = Vocab("bertVocab")
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    vocabLines = open(os.path.join(current_dir, "bert-base-uncased-vocab.txt"), "rb").readlines()
    for i in range(len(vocabLines)):
        voc.add_word(vocabLines[i].decode("utf-8").replace("\n", "").strip())
    vocabCounts = open(os.path.join(current_dir, "wordCounts.txt"), "r", encoding="utf-8").readlines()
    vocabCounts = [float(i.replace('\n', '')) for i in vocabCounts]
    for i in range(len(voc.word2count)):
        voc.word2count[list(voc.word2count.keys())[i]] = vocabCounts[i]
    if additional_tokens is not None:
        for i in range(len(additional_tokens)):
            voc.add_word(additional_tokens[i])
    return(voc)

# Make a vocab containing the alphabet and puncuation
def getAlphabetVocab(additional_tokens=None, add_default_tokens=True):
    assert additional_tokens is None or (isinstance(additional_tokens, list) and isinstance(additional_tokens[0], str))
    from string import ascii_lowercase
    voc = Vocab("alphabetVocab", add_default_tokens=add_default_tokens)
    # for letter in ascii_lowercase: # Add letters
    #     voc.add_word(str(letter))
    # voc.add_list([".", "'", "!", "?", " ", ",", ";", "-"]) # Add other symbols
    voc.add_list([
    "'",  # 0
    " ",  # 1
    "a",  # 2
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",  # 27
    "_",  # 28, blank
    ])
    if additional_tokens is not None:
        for i in range(len(additional_tokens)):
            voc.add_word(additional_tokens[i])
    return voc