spacyNLP = None
tokenizer = None

# SPACY SUPPORT FUNCTIONS
def check_spacy():
    load_spacy_small() # defaults to spacy small model if none is previously selected

def load_spacy_small(overwrite=False):
    if spacyNLP == None or overwrite:
        import spacy
        spacyNLP = spacy.load("en_core_web_sm")

def load_spacy_med(overwrite=False):
    if spacyNLP == None or overwrite:
        import spacy
        spacyNLP = spacy.load("en_core_web_md")

def load_spacy_large(overwrite=False):
    if spacyNLP == None or overwrite:
        import spacy
        spacyNLP = spacy.load("en_core_web_lg")

# HUGGINGFACE SUPPORT FUNCTIONS
def check_huggingface():
    if tokenizer = None:
        load_bert_wordpiece()

def load_bert_wordpiece(lowercase=True, overwrite=False):
    if tokenizer == None or overwrite:
        from tokenizers import BertWorkPieceTokenizer
        tokenizer = BertWordPieceTokenizer("bert-base-uncased-vocab.txt", lowercase=lowercase)

def load_sentencepiece(lowercase=True, overwrite=False):
    if tokenizer == None or overwrite:
        from tokenizers import SentencePieceBPETokenizer
        tokenizer = SentencePieceBPETokenizer()


# TOKENIZING FUNCTIONS
def tokenize(sentence, text=False):
    # Check if there is no initialized tokenizer
    if tokenizer == None and spacyNLP == None:
        check_huggingface() # Favor huggingface since it is faster and more flexible

    # Do tokenization based on avaliable tokenizer
    if tokenizer != None:
        if text:
            return(tokenizer.encode(sentence).tokens)
        else:
            return(tokenizer.encode(sentence))
    else if spacyNLP != None:
        if text:
            return [token.text for token in spacyNLP(sentence)]
        else:
            return spacyNLP(sentence)
