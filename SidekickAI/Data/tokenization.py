# Sidekick Tokenization v1.1
import os, copy
tokenizer = None
loaded_spacy_model = None # We have to represent the spacy model as a string...


# MAIN TOKENIZING FUNCTIONS
def tokenize_spacy(sentence, full_token_data=False, spacy_model=None):
     '''General Spacy tokenization function.\n
    Inputs:
        sentence: any possibly jagged collection of strings / tuples of strings (for BERT seperation)
        full_token_data: [default: false] return the full "tokenizers" module token object or simply the token string
        spacy_model: [default: None] the string of the spacy model ['en_core_web_sm', 'en_core_web_md', "en_core_web_lg'], defaults to medium
    Outputs:
        sentence: the same possibly jagged collection of inputs, now tokenized'''
    global tokenizer
    # Check if the initialized tokenizer is a spacy tokenizer
    import en_core_web_sm, en_core_web_md, en_core_web_lg
    models, model_names = [en_core_web_sm, en_core_web_md, en_core_web_lg], ["en_core_web_sm", "en_core_web_md", "en_core_web_lg"]
    assert spacy_model in model_names or spacy_model is None, "Unknown spacy model version: " + spacy_model
    if (loaded_spacy_model != spacy_model and spacy_model is not None) or loaded_spacy_model is None: tokenizer = models[model_names.index(spacy_model)].load() if spacy_model is not None else models[1].load()
    
    if isinstance(sentence, str): # Tokenize immediately
        return tokenizer(sentence) if full_token_data else [token.text for token in tokenizer(sentence)]
    
    # Traverse list, return linearized list of strings/pairs and insert index numbers into the main list
    main_list, linear_list = extract_bottom_items(main_list=sentence, base_types=[str, tuple])

    # Tokenize
    linear_list = [[token if full_token_data else token.text for token in doc] for doc in tokenizer.pipe(linear_list)]

    # Add the linear list items back to the main list
    main_list = insert_bottom_items(main_list=main_list, linear_list=linear_list)

    return main_list


def tokenize_wordpiece(sentence, special_tokens=False, full_token_data=False, lowercase=True):
    '''General WordPiece tokenization function.\n
    Inputs:
        sentence: any possibly jagged collection of strings / tuples of strings (for BERT seperation)
        special_tokens: [default: false] add in special BERT tokens such as CLS and SEP
        full_token_data: [default: false] return the full "tokenizers" module token object or simply the token string
        lowercase: [default: true] lowercase everything
    Outputs:
        sentence: the same possibly jagged collection of inputs, now tokenized'''

    global tokenizer
    from tokenizers import BertWordPieceTokenizer

    # Check if the initialized tokenizer is a wordpiece tokenizer
    if not isinstance(tokenizer, BertWordPieceTokenizer):
        tokenizer = BertWordPieceTokenizer(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bert-base-uncased-vocab.txt"), lowercase=lowercase)
    
    if isinstance(sentence, str): # Tokenize string right away
        if full_token_data:
            return tokenizer.encode(sentence) if special_tokens else tokenizer.encode(sentence)[1:-1]
        else:
            return tokenizer.encode(sentence).tokens if special_tokens else tokenizer.encode(sentence).tokens[1:-1]
    
    # Traverse list, return linearized list of strings/pairs and insert index numbers into the main list
    main_list, linear_list = extract_bottom_items(main_list=sentence, base_types=[str, tuple])
    # Check if we should add special tokens
    special_tokens = special_tokens or any([isinstance(x, tuple) for x in linear_list])

    # Tokenize the linear list
    if full_token_data:
        linear_list = [doc if special_tokens else doc[1:-1] for doc in tokenizer.encode_batch(linear_list)]
    else:
        linear_list = [doc.tokens if special_tokens else doc.tokens[1:-1] for doc in tokenizer.encode_batch(linear_list)]

    # Add the linear list items back to the main list
    main_list = insert_bottom_items(main_list=main_list, linear_list=linear_list)

    return main_list

# RECURSIVE TREE FUNCTIONS
def extract_bottom_items(main_list, base_types, linear_list=[]):
    assert list not in base_types, "List cannot be a base type!"
    # Check each element in list to see if it is another list, a string, or a tuple
    for i in range(len(main_list)):
        if type(main_list[i]) in base_types:
            # Replace item with index in the linear list
            linear_list.append(copy.copy(main_list[i]))
            main_list[i] = len(linear_list) - 1
        elif isinstance(main_list[i], list):
            # Run this function on the sublist
            main_list[i], linear_list = extract_bottom_items(main_list=main_list[i], linear_list=linear_list, base_types=base_types)
        else:
            # Throw type error
            raise Exception("Got an element of type " + str(type(main_list[i])) + " in the input! All base level types should be specified in base_types!")
        
    return main_list, linear_list

def insert_bottom_items(main_list, linear_list):
    # Loop through each element and replace it with it's linear list index if it is an int
    for i in range(len(main_list)):
        if isinstance(main_list[i], list): # Call this function on the sublist
            main_list[i] = insert_bottom_items(main_list=main_list[i], linear_list=linear_list)
        else: # Replace element with it's index
            main_list[i] = linear_list[main_list[i]]
    
    return main_list

# Tokenizes setence or list of sentences
def base_tokenize(sentence, full_token_data=False):
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
                return(tokenizer.encode(sentence).tokens[1:-1])
            
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


# UNTOKENIZATION FUNCTIONS
def untokenize_wordpiece(tokens, EOS_token=None):
    if tokens is None:
        return ""
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