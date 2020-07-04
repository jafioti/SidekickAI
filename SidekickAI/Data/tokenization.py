# Sidekick Tokenization v1.1
spacyNLP = None
tokenizer = None

def load_sentencepiece(lowercase=True, overwrite=False):
    global tokenizer
    if tokenizer is None or overwrite:
        from tokenizers import SentencePieceBPETokenizer
        tokenizer = SentencePieceBPETokenizer()


# MAIN TOKENIZING FUNCTIONS
def tokenize_spacy(sentence, full_token_data=False, spacy_model="en_core_web_md"):
    global tokenizer
    # Check if the initialized tokenizer is a spacy tokenizer
    import en_core_web_sm, en_core_web_md, en_core_web_lg
    models, model_names = [en_core_web_sm, en_core_web_md, en_core_web_lg], ["en_core_web_sm", "en_core_web_md", "en_core_web_lg"]
    assert spacy_model in model_names, "Unknown spacy model version: " + spacy_model
    if not isinstance(tokenizer, models[model_names.index(spacy_model)]): tokenizer = models[model_names.index(spacy_model)].load()
    
    if isinstance(sentence, str): # Tokenize immediately
        return tokenizer(sentence) if full_token_data else [token.text for token in tokenizer(sentence)]
    
    # Traverse list, return linearized list of strings/pairs and insert index numbers into the main list
    linear_list, main_list = extract_bottom_items(main_list=sentence, base_types=[str, tuple])

    # Tokenize
    linear_list = [[token if full_token_data else token.text for token in doc] for doc in tokenizer(linear_list)]

    # Add the linear list items back to the main list
    main_list = insert_bottom_items(main_list=main_list, linear_list=linear_list)

    return main_list


def tokenize_wordpiece(sentence, special_tokens=False, full_token_data=False):
    global tokenizer
    from tokenizers import BertWordPieceTokenizer
    if tokenize_pair: special_tokens = True

    # Check if the initialized tokenizer is a wordpiece tokenizer
    if not isinstance(tokenizer, BertWordPieceTokenizer):
        tokenizer = BertWordPieceTokenizer(os.path.join(current_dir, "bert-base-uncased-vocab.txt"), lowercase=lowercase)
    
    if isinstance(sentence, str): # Tokenize string right away
        if full_token_data:
            return tokenizer.encode(sentence) if special_tokens else tokenizer.encode(sentence)[1:-1]
        else:
            return tokenizer.encode(sentence).tokens if special_tokens else tokenizer.encode(sentence).tokens[1:-1]
    
    # Traverse list, return linearized list of strings/pairs and insert index numbers into the main list
    linear_list, main_list = extract_bottom_items(main_list=sentence, base_types=[str, tuple])

    # Check if we should add special tokens
    special_tokens = special_tokens or any([isinstance(x, tuple) for x in linear_list])

    # Tokenize the linear list
    if full_token_data:
        linear_list = [doc if special_tokens else doc[1:-1] for doc in tokenizer.encode_batch(linear_list, add_special_tokens=special_tokens)]
    else:
        linear_list = [doc.tokens if special_tokens else doc.tokens[1:-1] for doc in tokenizer.encode_batch(linear_list, add_special_tokens=special_tokens)]

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
            linear_list.append(main_list[i])
            main_list[i] = len(linear_list) - 1
        elif isinstance(main_list[i], list):
            # Run this function on the sublist
            main_list[i], linear_list = extract_bottom_items(main_list=main_list[i], linear_list=linear_list)
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