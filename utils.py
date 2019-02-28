import numpy as np

def pad_one_more(A):
    if len(A.shape) == 2:
        return np.pad(A, ((0,1), (0,1)), 'edge')
    else:
        return np.pad(A, ((0,1), (0,1), (0, 0)), 'edge')

def remove_pad_one_more(A):
    return A[:-1, :-1, ...]