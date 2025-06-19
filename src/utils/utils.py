"""
Source: STHN: utils.py
URL: https://github.com/celi52/STHN/blob/main/utils.py
"""

import random
import numpy as np
import torch
import torch_sparse
from tgb.linkproppred.evaluate import Evaluator


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


def evaluate_mrr(pred, neg_samples):
    metric = "mrr"
    # 计算每组的样本数量
    split = len(pred) // (neg_samples + 1)

    # 划分正样本和负样本
    num_groups = neg_samples + 1
    y_pred_pos_list = []
    y_pred_neg_list = []
    for i in range(num_groups - 1):
        y_pred_pos_list.append(pred[i * split : (i + 1) * split])
        y_pred_neg_list.append(pred[(num_groups - 1) * split :])

    evaluator = Evaluator(name="thgl-software")

    metric_values = []
    for y_pred_pos, y_pred_neg in zip(y_pred_pos_list, y_pred_neg_list):
        y_pred_pos = np.array(y_pred_pos.detach().cpu().numpy())
        y_pred_neg = np.array(y_pred_neg.detach().cpu().numpy())
        input_dict = {
            "y_pred_pos": y_pred_pos,
            "y_pred_neg": y_pred_neg,
            "eval_metric": [metric],
        }
        metric_value = evaluator.eval(input_dict)[metric]
        metric_values.append(metric_value)

    # 计算平均指标值
    average_metric = np.mean(metric_values)
    return average_metric
