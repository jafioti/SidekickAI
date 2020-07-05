import torch

def weighted_avg(x, weights):
    """Takes a weighted average of input vectors given the weights\n
    Inputs:
        x = (batch size, seq len, hidden dim)
        weights = (batch size, seq len)
    Returns:
        avg = (batch size, hidden dim)
    """
    return weights.unsqueeze(1).bmm(x).squeeze(1)

def uniform_avg(x):
    """Takes a uniform average of input vectors\n
    Inputs:
        x = (batch size, seq len, hidden dim)
    Returns:
        avg = (batch size, hidden dim)
    """
    seq_len = x.shape[1]
    return torch.sum(x, dim=1) / seq_len