"""
Source: STHN: utils.py
URL: https://github.com/celi52/STHN/blob/main/utils.py
"""

from math import e
import random
import numpy as np
import torch
import torch_sparse
from tgb.linkproppred.evaluate import Evaluator
import pandas as pd


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


def get_edge_type_distribution(dataset):
    df_edges = pd.read_csv(f"tgb/DATA/{dataset.replace('-','_')}/{dataset}_edgelist.csv")
    df_nodes = pd.read_csv(f"tgb/DATA/{dataset.replace('-','_')}/{dataset}_nodetype.csv")
    # Create a mapping from node id to node type
    node_type_map = dict(zip(df_nodes.iloc[:, 0], df_nodes.iloc[:, 1]))

    # Initialize dictionary to store edge type distribution
    edge_type_distribution = {}

    # Iterate through edges to build distribution
    for _, row in df_edges.iterrows():
        src_node = row.iloc[1]  # source node
        dst_node = row.iloc[2]  # destination node
        edge_type = row.iloc[3]  # edge type
        
        # Get node types
        src_type = node_type_map.get(src_node)
        dst_type = node_type_map.get(dst_node)
        
        # Create tuple key
        key = (src_type, dst_type)
        
        # Add edge type to the list for this node type pair
        if key not in edge_type_distribution:
            edge_type_distribution[key] = []
        if edge_type not in edge_type_distribution[key]:
            edge_type_distribution[key].append(edge_type)
    return edge_type_distribution

EDGE_TYPE_DISTRIBUTION = {
    "thgl-forum-subset": get_edge_type_distribution("thgl-forum-subset"),
    "thgl-myket-subset": get_edge_type_distribution("thgl-myket-subset"),
    "thgl-software-subset": get_edge_type_distribution("thgl-software-subset"),
    "thgl-github-subset": get_edge_type_distribution("thgl-github-subset"),
}
def get_node_type_dict(dataset):
    df_nodes = pd.read_csv(f"tgb/DATA/{dataset.replace('-','_')}/{dataset}_nodetype.csv")
    # 创建一个字典，键为第一列的值，值为第二列的值
    node_type_dict = dict(zip(df_nodes.iloc[:, 0], df_nodes.iloc[:, 1]))
    return node_type_dict


NODE_TYPE_DICT = {
    "thgl-forum-subset": get_node_type_dict("thgl-forum-subset"),
    "thgl-myket-subset": get_node_type_dict("thgl-myket-subset"),
    "thgl-software-subset": get_node_type_dict("thgl-software-subset"),
    "thgl-github-subset": get_node_type_dict("thgl-software-subset"),
}

def get_edge_types_poss_mask(dataset, src_nodes, dst_nodes):
    """
    获取边类型的可能性掩码
    :param dataset: 数据集对象
    :param src_nodes: 源节点列表
    :param dst_nodes: 目标节点列表
    :return: 可能性掩码张量
    """
    
    if "forum" in dataset:
        num_edge_types = 2  
    elif "myket" in dataset:
        num_edge_types = 2
    elif "github" in dataset:
        num_edge_types = 14
    elif "software" in dataset:
        num_edge_types = 14
    edge_type_distribution = EDGE_TYPE_DISTRIBUTION[dataset]
    node_type_dict = NODE_TYPE_DICT[dataset]

    # 批量获取源节点和目标节点类型
    src_types = np.array([node_type_dict[node] for node in src_nodes])
    dst_types = np.array([node_type_dict[node] for node in dst_nodes])
    
    # 创建所有唯一的(src_type, dst_type)组合
    node_type_pairs = list(zip(src_types, dst_types))
    unique_pairs, inverse_indices = np.unique(node_type_pairs, axis=0, return_inverse=True)
    
    # 预计算每个唯一组合对应的边类型掩码
    unique_masks = np.zeros((len(unique_pairs), num_edge_types), dtype=bool)
    for i, (src_type, dst_type) in enumerate(unique_pairs):
        edge_types = edge_type_distribution.get((src_type, dst_type), [])
        if edge_types:
            unique_masks[i, edge_types] = True
    
    # 根据inverse_indices映射回原始顺序
    poss_mask = unique_masks[inverse_indices]
    res_poss_mask = torch.from_numpy(poss_mask)

    return res_poss_mask
    



    
