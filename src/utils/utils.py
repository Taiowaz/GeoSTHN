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


def generate_short_dir_name(args):
    # 简化参数名，生成短目录名
    dir_name_parts = [
        f"ep{args.num_epoch}",
        f"s{args.seed}",
        f"nr{args.num_run}",
        f"ds{args.dataset}",
        f"gpu{args.use_gpu}",
        f"dev{args.device}",
        f"bs{args.batch_size}",
        f"me{args.max_edges}",
        f"et{args.num_edgeType}",
        f"lr{args.lr}",
        f"wd{args.weight_decay}",
        f"pc{args.predict_class}",
        f"ws{args.window_size}",
        f"do{args.dropout}",
        f"ns{args.neg_samples}",
        f"ens{args.extra_neg_samples}",
        f"nn{args.num_neighbors}",
        f"cef{args.channel_expansion_factor}",
        f"sh{args.sampled_num_hops}",
        f"td{args.time_dims}",
        f"hd{args.hidden_dims}",
        f"nl{args.num_layers}",
        f"cdl{args.check_data_leakage}",
        f"in{args.ignore_node_feats}",
        f"ne{args.node_feats_as_edge_feats}",
        f"ie{args.ignore_edge_feats}",
        f"on{args.use_onehot_node_feats}",
        f"nt{args.use_node_type_as_feats}",
        f"tf{args.use_type_feats}",
        f"gs{args.use_graph_structure}",
        f"stg{args.structure_time_gap}",
        f"shp{args.structure_hops}",
        f"nc{args.use_node_cls}",
        f"cs{args.use_cached_subgraph}",
    ]
    return "_".join(dir_name_parts)
