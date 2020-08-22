# Sidekick Batching v1.0
import itertools, random
from torch import LongTensor, BoolTensor, Tensor
import SidekickAI.Data.tokenization
# This file holds functions to convert sentence pair batches to structured tensors to feed into models

def pad_mask(input_batch, pad_value):
    '''Makes binary (0, 1) matrix for batch depending on if token is padding (0 if so, 1 if not)'''
    m = []
    for i, seq in enumerate(input_batch):
        m.append([])
        for token in seq:
            if token != pad_value:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

def pad_batch(input_batch, fillvalue):
    '''Pads all inputs to longest input'''
    return list(itertools.zip_longest(*input_batch, fillvalue=fillvalue))

# Makes batches from a raw list of examples
#def batch(dataset):

# Returns padded sequence tensor, lengths, and pad mask
def batch_to_train_data(indexes_batch, PAD_token, return_lengths=False, return_pad_mask=False):
    '''
    Returns training data for a given batch.
        Inputs:
            indexes_batch (list): A list of lists of token indexes
            PAD_token (int): An index of the pad token
            *return_lengths (bool): Whether or not to return the lengths of the unpadded sequences [default: False]
            *return_pad_mask (bool): Whether or not to return a pad mask over all the padding [default: False]

        Returns:
            output_tensor (tensor: (seq len, batch size)): The padded tensor to input to the model
            *lengths (tensor: (batch size)): A tensor specifying the actual lengths of each sequence without padding
            *mask (tensor: (batch size, seq len)): A binary tensor specifying if the current position is a pad token or not
    '''
    # Pad inputs to longest length
    padList = pad_batch(indexes_batch, fillvalue=PAD_token)
    padVar = LongTensor(padList)
    return_list = [padVar]
    if return_lengths:
        # Get lengths of each sentence in batch
        return_list.append(Tensor([len(indexes) for indexes in indexes_batch]))
    if return_pad_mask:
        # Get mask over all the pad tokens
        return_list.append(BoolTensor(pad_mask(padList, pad_value=PAD_token)))
    return tuple(return_list) if len(return_list) > 1 else return_list[0]

def filter_by_length(*lists, max_length):
    '''Filters list or lists by a max length and returns the onces under the max'''
    lists = [*lists]
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
    '''
    Shuffle multiple lists in the same way
        Inputs:
            lists (lists): The lists to be shuffled
        Outputs:
            lists (lists): The shuffled lists
        Usage:
            list1, list2, list3 = shuffle_lists(list1, list2, list3)
    '''
    zipped_lists = list(zip(*lists))
    random.shuffle(zipped_lists)
    return zip(*zipped_lists)

def sort_lists_by_length(sorting_list, *other_lists, sorting_function=None, longest_first=False):
    '''
    Sort multiple lists by the lengths of the first list of lists
        Inputs:
            sorting_list (list of lists): The list of lists to be used when sorting
            other_lists (lists): The other lists to be sorted in the same way
            sort_function (function): A function determining the way to find the length of the example
            *longest_first (bool): Sort with the longest coming first [default: False]
        Outputs:
            lists (lists): The sorted lists
        Usage:
            list1, list2, list3 = sort_lists_by_length(list1, list2, list3)
    '''
    is_other_lists = other_lists is not None and len(other_lists) > 0
    zipped_lists = list(zip(sorting_list, *other_lists)) if is_other_lists else sorting_list
    zipped_lists.sort(reverse=longest_first, key=(lambda x: len(x[0])) if sorting_function is None else sorting_function)
    return zip(*zipped_lists) if is_other_lists else zipped_lists