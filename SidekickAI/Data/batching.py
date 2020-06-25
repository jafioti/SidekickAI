# Sidekick Batching v1.0
import itertools, random
from torch import LongTensor, BoolTensor, Tensor
import SidekickAI.Data.tokenization
# This file holds functions to convert sentence pair batches to structured tensors to feed into models

# Makes binary (0, 1) matrix for batch depending on if token is padding (0 if so, 1 if not)
def pad_mask(input_batch, pad_value):
    m = []
    for i, seq in enumerate(input_batch):
        m.append([])
        for token in seq:
            if token != pad_value:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

# Pads all inputs to longest input
def pad_batch(input_batch, fillvalue):
    return list(itertools.zip_longest(*input_batch, fillvalue=fillvalue))

# Returns padded input sequence tensor and lengths
def input_batch_to_train_data(indexes_batch, PAD_token, return_lengths=False):
    # Pad inputs to longest length
    padList = pad_batch(indexes_batch, fillvalue=PAD_token)
    padVar = LongTensor(padList)
    if return_lengths:
        # Get lengths of each sentence in batch
        lengths = Tensor([len(indexes) for indexes in indexes_batch])
        return padVar, lengths
    else:
        return padVar

# Returns padded target sequence tensor, padding mask, and max target length
def output_batch_to_train_data(indexes_batch, PAD_token, return_mask=False, return_max_target_length=False):
    if return_max_target_length:
        # Get max length
        max_target_len = max([len(indexes) for indexes in indexes_batch])
    # Pad batch
    padList = pad_batch(indexes_batch, PAD_token)
    padVar = LongTensor(padList)
    return_set = [padVar]
    if return_mask:
        # Get binary pad mask
        mask = pad_mask(padList, pad_value=PAD_token)
        mask = BoolTensor(mask)
        return_set.append(mask)
    if return_max_target_length:
        return_set.append(max_target_len)
    if len(return_set) > 1:
        return tuple(return_set)
    else:
        return return_set[0]

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

# Shuffles multiple lists of the same length in the same ways
def shuffle_lists(*lists):
    zipped_lists = list(zip(*lists))
    random.shuffle(zipped_lists)
    return zip(*zipped_lists)

# Sorts multiple lists by the lengths of the first list
def sort_lists_by_length(sorting_list, *other_lists, longest_first=False):
    zipped_lists = list(zip(sorting_list, *other_lists))
    def sorting_function(e): return len(e[0])
    zipped_lists.sort(reverse=longest_first, key=sorting_function)
    return zip(*zipped_lists)