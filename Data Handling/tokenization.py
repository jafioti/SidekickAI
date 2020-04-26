# Sidekick Tokenization v1.1
spacyNLP = None
tokenizer = None

# SPACY SUPPORT FUNCTIONS
def check_spacy():
    load_spacy_small() # defaults to spacy small model if none is previously selected

def load_spacy_small(overwrite=False):
    global spacyNLP
    if spacyNLP is None or overwrite:
        import spacy
        spacyNLP = spacy.load("en_core_web_sm")

def load_spacy_med(overwrite=False):
    global spacyNLP
    if spacyNLP is None or overwrite:
        import spacy
        spacyNLP = spacy.load("en_core_web_md")

def load_spacy_large(overwrite=False):
    global spacyNLP
    if spacyNLP is None or overwrite:
        import spacy
        spacyNLP = spacy.load("en_core_web_lg")

# HUGGINGFACE SUPPORT FUNCTIONS
def check_huggingface():
    global tokenizer
    if tokenizer is None:
        load_bert_wordpiece()

def load_bert_wordpiece(lowercase=True, overwrite=False):
    global tokenizer
    if tokenizer is None or overwrite:
        from tokenizers import BertWordPieceTokenizer
        tokenizer = BertWordPieceTokenizer("bert-base-uncased-vocab.txt", lowercase=lowercase)

def load_sentencepiece(lowercase=True, overwrite=False):
    global tokenizer
    if tokenizer is None or overwrite:
        from tokenizers import SentencePieceBPETokenizer
        tokenizer = SentencePieceBPETokenizer()


# MAIN TOKENIZING FUNCTION
def tokenize(sentence, full_token=False):
    global tokenizer, spacyNLP
    # Check if there is no initialized tokenizer
    if tokenizer is None and spacyNLP is None:
        check_huggingface() # Favor huggingface since it is faster and more flexible
    
    if isinstance(sentence, list):
        # Tokenize multi level list with recursive function
        return(traverseList(sentence))
    else:
        return(base_tokenize(sentence, full_token=full_token))

# Recursive function to get lengths of jagged list
def traverseList(sentence):
    currentLevel = []
    for i in range(len(sentence)):
        if isinstance(sentence[i], list):
            nextLevel = traverseList(sentence[i])
            if nextLevel is None:
                currentLevel.append(base_tokenize(sentence[i]))
            else:
                currentLevel.append(nextLevel)
        else:
            return base_tokenize(sentence)
    return currentLevel

# Tokenizes setence or list of sentences
def base_tokenize(sentence, full_token = False):
    # Do tokenization based on avaliable tokenizer
    if tokenizer is not None:
        if isinstance(sentence, list):
            # Batch
            if full_token:
                return(tokenizer.encode_batch(sentence))
            else:
                return([singleSentence.tokens[1:-1] for singleSentence in tokenizer.encode_batch(sentence)])
        else:
            # Single
            if full_token:
                return(tokenizer.encode(sentence))
            else:
                return(tokenizer.encode(sentence).tokens)
            
    elif spacyNLP is not None:
        if isinstance(sentence, list):
            # Batch
            if full_token:
                return [spacyNLP(sentence[i]) for i in range(len(sentence))]
            else:
                return [[token.text for token in spacyNLP(sentence[i])] for i in range(len(sentence))]
        else:
            # Single
            if full_token:
                return spacyNLP(sentence)
            else:
                return [token.text for token in spacyNLP(sentence)]

def untokenize_wordpiece(tokens, EOS_token):
    if len(tokens) == 0:
        return ""
    finalString = ""
    punctuation = [".", "?", "!", ",", "'", '"']
    for i in range(len(tokens)):
        if tokens[i] != EOS_token and tokens[i] != "[CLS]":
            if "##" not in tokens[i] and tokens[i] not in punctuation:
                finalString += " " + tokens[i]
            else:
                finalString += tokens[i].replace("##", "")
    return(finalString.replace("[SEP]", "").replace("EOS", ""))