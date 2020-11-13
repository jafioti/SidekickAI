# Sidekick Tokenization v1.1
import os, copy
tokenizer = None
loaded_spacy_model = None # We have to represent the spacy model as a string...


# MAIN TOKENIZING FUNCTIONS
def tokenize_spacy(sentence, full_token_data=False, spacy_model=None):
    sentence = copy.deepcopy(sentence)
    '''General Spacy tokenization function.\n
    Inputs:
        sentence: any possibly jagged collection of strings / tuples of strings (for BERT seperation)
        full_token_data: [default: false] return the full "tokenizers" module token object or simply the token string
        spacy_model: [default: None] the string of the spacy model ['en_core_web_sm', 'en_core_web_md', "en_core_web_lg'], defaults to medium
    Outputs:
        sentence: the same possibly jagged collection of inputs, now tokenized'''
    global tokenizer, loaded_spacy_model
    # Check if the initialized tokenizer is a spacy tokenizer
    import en_core_web_sm, en_core_web_md, en_core_web_lg
    models, model_names = [en_core_web_sm, en_core_web_md, en_core_web_lg], ["en_core_web_sm", "en_core_web_md", "en_core_web_lg"]
    assert spacy_model in model_names or spacy_model is None, "Unknown spacy model version: " + spacy_model
    if (loaded_spacy_model != spacy_model and spacy_model is not None) or loaded_spacy_model is None:
        tokenizer = models[model_names.index(spacy_model)].load() if spacy_model is not None else models[1].load()
        loaded_spacy_model = spacy_model if spacy_model is not None else "en_core_web_md"
    
    if isinstance(sentence, str): # Tokenize immediately
        return tokenizer(sentence) if full_token_data else [token.text for token in tokenizer(sentence)]
    
    # Traverse list, return linearized list of strings/pairs and insert index numbers into the main list
    main_list, linear_list = extract_bottom_items(main_list=sentence, base_types=[str])

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
    
    if isinstance(sentence, str) or isinstance(sentence, tuple):
        tokenized = tokenizer.encode(sentence) if isinstance(sentence, str) else tokenizer.encode(sentence[0], sentence[1])
        if not full_token_data: tokenized = tokenized.tokens
        if not special_tokens: tokenized = tokenized[1:-1]
        return tokenized
    
    # Traverse list, return linearized list of strings/pairs and insert index numbers into the main list
    main_list, linear_list = extract_bottom_items(main_list=sentence, base_types=[str, tuple])
    del sentence
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

def tokenize_moses(sentence, lowercase=True):
    sentence = copy.deepcopy(sentence)
    '''General Moses tokenization function.\n
    Inputs:
        sentence: any possibly jagged collection of strings
        lowercase: [default: true] lowercase everything
    Outputs:
        sentence: the same possibly jagged collection of inputs, now tokenized'''
    global tokenizer
    from sacremoses import MosesTokenizer
    if not isinstance(tokenizer, MosesTokenizer): tokenizer = MosesTokenizer(lang="en")

    if isinstance(sentence, str): # Tokenize string right away
        return tokenizer.tokenize(sentence.lower()) if lowercase else tokenizer.tokenize(sentence)
    
    # Traverse list, return linearized list of strings/pairs and insert index numbers into the main list
    main_list, linear_list = extract_bottom_items(main_list=sentence, base_types=[str])

    # Tokenize the linear list
    for i in range(len(linear_list)):
        linear_list[i] = tokenizer.tokenize(linear_list[i].lower()) if lowercase else tokenizer.tokenize(linear_list[i])

    # Add the linear list items back to the main list
    main_list = insert_bottom_items(main_list=main_list, linear_list=linear_list)

    return main_list

def tokenize_alphabet(sentence, lowercase=True):
    ''' Tokenizes sentence to the alphabet vocab'''
    sentence = copy.deepcopy(sentence)
    if type(sentence) == str: return [char for char in (sentence.lower() if lowercase else sentence)]

    # Traverse list, return linearized list of strings/pairs and insert index numbers into the main list
    main_list, linear_list = extract_bottom_items(main_list=sentence, base_types=[str])

    # Tokenize the linear list
    for i in range(len(linear_list)):
        linear_list[i] = [char for char in (linear_list[i].lower() if lowercase else linear_list[i])]

    # Add the linear list items back to the main list
    main_list = insert_bottom_items(main_list=main_list, linear_list=linear_list)
    return main_list

def tokenize_custom(sentence, tokenization_function, lowercase=True):
    sentence = copy.deepcopy(sentence)
    '''Custom tokenization function. Allows for custom functions to take in a string and output a list of tokens\n
    Inputs:
        sentence: any possibly jagged collection of strings
        tokenization_function (function): Tokenization function that takes in a string and outputs a list of tokens
        lowercase: [default: true] lowercase everything
    Outputs:
        sentence: the same possibly jagged collection of inputs, now tokenized'''
    if isinstance(sentence, str): # Tokenize string right away
        return tokenization_function(sentence)
    
    # Traverse list, return linearized list of strings/pairs and insert index numbers into the main list
    main_list, linear_list = extract_bottom_items(main_list=sentence, base_types=[str])

    # Tokenize the linear list
    for i in range(len(linear_list)):
        linear_list[i] = tokenization_function(linear_list[i].lower()) if lowercase else tokenization_function(linear_list[i])

    # Add the linear list items back to the main list
    main_list = insert_bottom_items(main_list=main_list, linear_list=linear_list)

    return main_list

# RECURSIVE TREE FUNCTIONS
def extract_bottom_items(main_list, base_types, linear_list=None):
    if linear_list is None: linear_list = []
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

# UNTOKENIZATION FUNCTIONS
def untokenize_wordpiece(tokens):
    tokens = copy.deepcopy(tokens)
    '''Untokenization function for WordPiece tokens, also splits at [SEP] tokens, and removes [CLS] and EOS tokens\n
    Inputs:
        tokens: any possible non jagged list of tokens
    Returns:
        sentences: the same list of tokens, with the bottom level list replaced with a string'''
    if tokens is None:
        return ""
    if len(tokens) == 0:
        return ""
    for i in range(len(tokens)):
        if isinstance(tokens[i], list):
            tokens[i] = untokenize_wordpiece(tokens[i])
        elif i > 0:
            raise Exception("The list of tokens cannot be jagged!")
        else:
            finalString = [""]
            punctuation = [".", "?", "!", ",", "'", '"']
            for x in range(len(tokens)):
                if tokens[x] == "[SEP]":
                    tokens.append("")
                if tokens[x] != "EOS" and tokens[x] != "[CLS]":
                    if "##" not in tokens[x] and tokens[x] not in punctuation:
                        finalString[-1] += " " + tokens[x]
                    else:
                        finalString[-1] += tokens[x].replace("##", "")
            return finalString[0] if len(finalString) == 1 else finalString
    return tokens

def untokenize_moses(tokens):
    tokens = copy.deepcopy(tokens)
    '''Untokenization function for Moses tokens\n
    Inputs:
        tokens: any possible non jagged list of tokens
    Returns:
        sentences: the same list of tokens, with the bottom level list replaced with a string'''
    global tokenizer
    from sacremoses import MosesDetokenizer
    if not isinstance(tokenizer, MosesDetokenizer): tokenizer = MosesDetokenizer(lang="en") # We use the global tokenizer object since it will work well for the recursion
    if tokens is None:
        return ""
    if len(tokens) == 0:
        return ""
    for i in range(len(tokens)):
        if isinstance(tokens[i], list):
            tokens[i] = untokenize_moses(tokens[i])
        elif i > 0:
            raise Exception("The list of tokens cannot be jagged!")
        else:
            return tokenizer.detokenize(tokens)
    return tokens

def untokenize_alphabet(tokens):
    tokens = copy.deepcopy(tokens)
    '''Untokenization function for Alphabet tokens\n
    Inputs:
        tokens: any possible non jagged list of tokens
    Returns:
        sentences: the same list of tokens, with the bottom level list replaced with a string'''
    if tokens is None:
        return ""
    if len(tokens) == 0:
        return ""
    for i in range(len(tokens)):
        if isinstance(tokens[i], list):
            tokens[i] = untokenize_alphabet(tokens[i])
        elif i > 0:
            raise Exception("The list of tokens cannot be jagged!")
        else:
            return "".join(tokens)
    return tokens