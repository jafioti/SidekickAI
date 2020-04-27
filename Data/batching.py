# Sidekick Batching v1.0
import itertools
from torch import LongTensor, BoolTensor, Tensor
import Sidekick.Data.tokenization
# This file holds functions to convert sentence pair batches to structured tensors to feed into models

# Makes binary (0, 1) matrix for batch depending on if token is padding (0 if so, 1 if not)
def pad_mask(input_batch, pad_value):
    m = []
    for i, seq in enumerate(input_batch):
        m.append([])
        for token in seq:
            if token == pad_value:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

# Pads all inputs to longest input
def pad_batch(input_batch, fillvalue):
    return list(itertools.zip_longest(*input_batch, fillvalue=fillvalue))

# Recursivly gets indexes
def indexes_from_tokens(tokens, vocab):
    if tokens is None:
        return
    if len(tokens) == 0:
        return
    if isinstance(tokens[0], list):
        current = []
        for i in range(len(tokens)):
            nextLevel = indexes_from_tokens(tokens[i], vocab)
            if nextLevel is not None:
                current.append(nextLevel)
        return current
    else:
        return [vocab.word2index[word] for word in tokens] + [vocab.EOS_token]

# Returns padded input sequence tensor and lengths
def input_batch_to_train_data(indexes_batch, vocab, return_lengths=False):
    # Pad inputs to longest length
    padList = pad_batch(indexes_batch, fillvalue=vocab.PAD_token)
    padVar = LongTensor(padList)
    if return_lengths:
        # Get lengths of each sentence in batch
        lengths = Tensor([len(indexes) for indexes in indexes_batch])
        return padVar, lengths
    else:
        return padVar

# Returns padded target sequence tensor, padding mask, and max target length
def output_batch_to_train_data(indexes_batch, vocab, return_mask=False, return_max_target_length=False):
    if return_max_target_length:
        # Get max length
        max_target_len = max([len(indexes) for indexes in indexes_batch])
    # Pad batch
    padList = pad_batch(indexes_batch, vocab.PAD_token)
    padVar = LongTensor(padList)
    return_set = [padVar]
    if return_mask:
        # Get binary pad mask
        mask = pad_mask(padList, pad_value=vocab.PAD_token)
        mask = BoolTensor(mask)
        return_set.append(mask)
    if return_max_target_length:
        return_set.append(max_target_len)
    return tuple(return_set)

# Sort batch of inputs and outputs by length
def sort_pair_batch_by_length(pair_batch):
    pair_batch.sort(key=lambda x: len(x[0]), reverse=True)
    return(pair_batch)

# Filters list or lists by a max length
def filter_by_length(lists, max_length):
    if isinstance(lists[0], list):
        new_lists = [[] for i in range(len(lists))]
        # List of lists
        for i in range(len(lists[0])):
            too_long = False
            for x in range(len(lists)):
                if len(lists[x][i]) > max_length:
                    too_long = True
                    break
            if not too_long:
                for x in range(len(lists)):
                    new_lists[x].append(lists[x][i])
        return(tuple(new_lists))
    else:
        # Single list
        return([lists[i] for i in range(len(lists)) if len(lists[i]) <= max_length])