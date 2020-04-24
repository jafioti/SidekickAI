import itertools
from torch import LongTensor, BoolTensor, Tensor
# This file holds functions to convert sentence pair batches to structured tensors to feed into models

# Makes binary (0, 1) matrix for batch depending on if token is padding (0 if so, 1 if not)
def binary_pad_matrix(input_batch, pad_value):
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
def zeroPadding(input_batch, fillvalue):
    return list(itertools.zip_longest(*input_batch, fillvalue=fillvalue))

# Tokenizes and gets indexes for sentence
def indexesFromSentence(vocab, sentence, tokenizationFunction):
    return [vocab.word2index[word] for word in tokenizationFunction(sentence)] + [vocab.EOS_token]

# Returns padded input sequence tensor and lengths
def input_tensor(input_batch, vocab, tokenizationFunction, return_lengths=False):
    # Get indexes
    indexes_batch = [indexesFromSentence(vocab, sentence, tokenizationFunction) for sentence in input_batch]
    # Pad inputs to longest length
    padList = zeroPadding(indexes_batch, fillvalue=vocab.PAD_token)
    padVar = LongTensor(padList)
    if return_lengths:
        # Get lengths of each sentence in batch
        lengths = Tensor([len(indexes) for indexes in indexes_batch])
        return padVar, lengths
    else:
        return padVar

# Returns padded target sequence tensor, padding mask, and max target length
def output_tensor(input_batch, vocab, tokenizationFunction, return_mask=False, return_max_target_length=False):
    # Get indexes
    indexes_batch = [indexesFromSentence(vocab, sentence, tokenizationFunction) for sentence in input_batch]
    if return_max_target_length:
        # Get max length
        max_target_len = max([len(indexes) for indexes in indexes_batch])
    # Pad batch
    padList = zeroPadding(indexes_batch, vocab.PAD_token)
    padVar = LongTensor(padList)
    return_set = [padVar]
    if return_mask:
        # Get binary pad mask
        mask = binary_pad_matrix(padList, pad_value=vocab.PAD_token)
        mask = BoolTensor(mask)
        return_set.append(mask)
    if return_max_target_length:
        return_set.append(max_target_len)
    return tuple(return_set)

# Returns all items for a given batch of pairs
def pair_batch_to_training_data(vocab, pair_batch, tokenizationFunction, return_input_lengths=False, return_mask=False, return_max_target_length=False):
    # Sort batch by length
    pair_batch.sort(key=lambda x: len(tokenizationFunction(x[0])), reverse=True)
    # Split into input and output batch
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    # Get vectorized input and input lengths
    inp = input_tensor(input_batch, vocab, tokenizationFunction, return_lengths=return_input_lengths)
    # Get vectorized output, output mask, and max_target_len
    output = output_tensor(output_batch, vocab, tokenizationFunction, return_mask=return_mask, return_max_target_length=return_max_target_length)
    return inp + output # Input contains all requested input objects, output contains all requested output objects