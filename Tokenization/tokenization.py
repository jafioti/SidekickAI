import tokenizers

spacyNLP = None
tokenizer = None

# SPACY SUPPORT FUNCTIONS
def check_spacy():
    import spacy
    if spacyNLP == None:
        spacyNLP = spacy.load("en_core_web_sm") # defaults to spacy small model if none is previously selected

def load_spacy_small():
    import spacy
    if spacyNLP == None:
        spacyNLP = spacy.load("en_core_web_sm")

def load_spacy_med():
    import spacy
    if spacyNLP == None:
        spacyNLP = spacy.load("en_core_web_md")

def load_spacy_large():
    import spacy
    if spacyNLP == None:
        spacyNLP = spacy.load("en_core_web_lg")

# HUGGINGFACE SUPPORT FUNCTIONS
def check_huggingface():
    if tokenizer = None:
        load_bert_wordpiece()

def load_bert_wordpiece(lowercase=True):
    from tokenizers import BertWorkPieceTokenizer
    tokenizer = BertWordPieceTokenizer("bert-base-uncased-vocab.txt", lowercase=lowercase)

def load_sentencepiece():
    from tokenizers import SentencePieceBPETokenizer
    tokenizer = SentencePieceBPETokenizer()


# TOKENIZING FUNCTIONS
def spacy_tokenize(sentence, text=False):
    check_spacy()
    if text:
        return [token.text for token in spacyNLP(sentence)]
    else:
        return spacyNLP(sentence)

def tokenizeHuggingface(sentence, text=False):
    check_huggingface()
    if text:
        return(tokenizer.encode(sentence).tokens)
    else:
        return(tokenizer.encode(sentence))
