import torch

def weighted_avg(vectors, weights):
    """Takes a weighted average of input vectors given the weights\n
    Inputs:
        vectors = (batch size, num vectors, hidden dim) or (num vectors, hidden dim)
        weights = (batch size, num vectors)
    Returns:
        avg = (batch size, hidden dim)"""
    if len(vectors.shape) == 2:
        return torch.mm(weights, vectors)
    else:
        return torch.bmm(vectors.transpose(1, 2), weights.unsqueeze(-1)).squeeze(-1)

def uniform_avg(x):
    """Takes a uniform average of input vectors\n
    Inputs:
        x = (batch size, num vectors, hidden dim)
    Returns:
        avg = (batch size, hidden dim)
    """
    num_vectors = x.shape[1]
    return torch.sum(x, dim=1) / num_vectors

def batch_dot(x, y):
    '''Takes a batch of dot products and returns the outputs\n
    Inputs:
        x = (x batch size, vector dim)
        y = (y batch size, vector dim)
    Outputs:
        outputs = (x batch size, y batch size)'''
    return torch.mm(x, y.transpose(0, 1))

def batch_matrix_vector(x, y):
    '''Takes a batch of matrix-vector dot products\n
    Inputs:
        x = (batch size, seq len, vector dim)
        y = (batch size, vector dim)
    Outputs:
        outputs = (batch size, batch size)'''
    return torch.matmul(x, y.unsqueeze(-1)).squeeze(-1)