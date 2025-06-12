"""
Source: STHN: utils.py
URL: https://github.com/celi52/STHN/blob/main/utils.py
"""
import random
import numpy as np
import torch
import torch_sparse

# utility function
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def row_norm(adj_t):
    if isinstance(adj_t, torch_sparse.SparseTensor):
        # adj_t = torch_sparse.fill_diag(adj, 1)
        deg = torch_sparse.sum(adj_t, dim=1)
        deg_inv = 1.0 / deg
        deg_inv.masked_fill_(deg_inv == float("inf"), 0.0)
        adj_t = torch_sparse.mul(adj_t, deg_inv.view(-1, 1))
        return adj_t
