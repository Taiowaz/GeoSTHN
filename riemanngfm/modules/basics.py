import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from riemanngfm.modules.layers import ConstCurveLinear
from torch_scatter import scatter_sum, scatter_softmax


class HyperbolicStructureLearner(nn.Module):
    def __init__(self, manifold_H, manifold_S, in_dim, hidden_dim, out_dim, dropout=0.1):
        super(HyperbolicStructureLearner, self).__init__()
        self.manifold_H = manifold_H
        self.manifold_S = manifold_S
        self.tree_agg = CrossManifoldAttention(manifold_S, manifold_H, in_dim, hidden_dim, out_dim, dropout)

    # def forward(self, x_H, x_S, batch_tree):
    #     """
    #     Local Attention based on BFS tree structure inherit from a sub-graph.
    #     :param x_H: Hyperbolic representation of nodes
    #     :param batch_tree: a batch graph with tree-graphs from one graph.
    #     :return: New Hyperbolic representation of nodes.
    #     """
    #     num_seeds = len(batch_tree)
    #     node_labels = torch.arange(x_H.shape[0], device=x_H.device).repeat(num_seeds)
    #     x = x_H[node_labels]
    #     att_index = batch_tree.edge_index
    #     x = self.tree_agg(x_S[node_labels], x, x, edge_index=att_index)

    #     x_extend = torch.concat([x, x_H], dim=0)
    #     label_extend = torch.cat(
    #         [node_labels, torch.arange(x_H.shape[0], device=x_H.device)],
    #         dim=0)
    #     z_H = self.manifold_H.Frechet_mean(x_extend, keepdim=True, sum_idx=label_extend)
    #     return z_H
    def forward(self, x_H, x_S, batch_tree):
        """
        Local Attention based on BFS tree structure inherit from a sub-graph.
        :param x_H: Hyperbolic representation of nodes
        :param batch_tree: a batch graph with tree-graphs from one graph.
        :return: New Hyperbolic representation of nodes.
        """
        num_seeds = len(batch_tree)
        num_nodes = x_H.shape[0]
        
        # 确保 batch_tree.edge_index 中的索引在有效范围内
        edge_index = batch_tree.edge_index
        max_node_idx = edge_index.max().item() if edge_index.numel() > 0 else -1
        
        if max_node_idx >= num_nodes:
            # 如果索引超出范围，需要进行调整或过滤
            valid_mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
            edge_index = edge_index[:, valid_mask]
        
        # 创建节点标签，但要确保与 edge_index 兼容
        node_labels = torch.arange(num_nodes, device=x_H.device)
        
        # 扩展节点特征以匹配批次结构
        if num_seeds > 1:
            # 如果有多个种子，需要适当处理批次维度
            x_q = x_S[node_labels]
            x_k = x_H[node_labels] 
            x_v = x_H[node_labels]
        else:
            x_q = x_S
            x_k = x_H
            x_v = x_H
        
        x = self.tree_agg(x_q, x_k, x_v, edge_index=edge_index)
        
        # 使用 Frechet mean 聚合结果
        x_extend = torch.concat([x, x_H], dim=0)
        label_extend = torch.cat(
            [torch.arange(x.shape[0], device=x_H.device), 
             torch.arange(x_H.shape[0], device=x_H.device)],
            dim=0)
        z_H = self.manifold_H.Frechet_mean(x_extend, keepdim=True, sum_idx=label_extend)
        return z_H


class SphericalStructureLearner(nn.Module):
    """
    in_dim = out_dim
    """
    def __init__(self, manifold_H, manifold_S, in_dim, hidden_dim, out_dim, dropout=0.1):
        super(SphericalStructureLearner, self).__init__()
        self.manifold_H = manifold_H
        self.manifold_S = manifold_S
        self.cross_attention = CrossManifoldAttention(manifold_H, manifold_S, in_dim, hidden_dim, out_dim, dropout)
        self.res_lin = nn.Linear(out_dim, out_dim)

    def forward(self, x_H, x_S, data):
        """
        :param x_H: Hyperbolic representation of nodes
        :param x_S: Sphere representation of nodes
        :param data: a graph or mini-batched graph
        :return: New sphere representation of nodes.
        """
        att_index = data.edge_index
        x = self.cross_attention(x_H, x_S, x_S, edge_index=att_index)
        z_S = self.manifold_S.expmap(x, self.manifold_S.proju(x, self.res_lin(x_S)))
        return z_S

# 🆕 NEW: 为“星型图”创建一个专属的新学习器模块
class StarStructureLearner(nn.Module):
    """
    一个专门用于学习星型图（中心化结构）的模块。
    它在双曲空间中运作，并使用与HyperbolicStructureLearner相同的注意力机制。
    """
    def __init__(self, manifold_H, manifold_S, in_dim, hidden_dim, out_dim, dropout=0.1):
        super(StarStructureLearner, self).__init__()
        self.manifold_H = manifold_H
        self.manifold_S = manifold_S
        # 我们可以复用强大的跨流形注意力机制
        self.star_agg = CrossManifoldAttention(manifold_S, manifold_H, in_dim, hidden_dim, out_dim, dropout)

    def forward(self, x_H, x_S, batch_star):
        # 这里的逻辑与 HyperbolicStructureLearner 非常相似
        num_graphs = batch_star.num_graphs
        
        # 扩展节点特征以匹配批次结构
        node_labels = torch.arange(x_H.shape[0], device=x_H.device).repeat(num_graphs)
        x_k_v = x_H[node_labels] # Key and Value from Hyperbolic space
        x_q = x_S[node_labels]   # Query from Spherical space
        
        # 使用注意力机制聚合星型图的边
        x = self.star_agg(x_q, x_k_v, x_k_v, edge_index=batch_star.edge_index)

        # 使用 Frechet mean 聚合结果，得到更新后的节点表示
        x_extend = torch.cat([x, x_H], dim=0)
        label_extend = torch.cat(
            [node_labels, torch.arange(x_H.shape[0], device=x_H.device)],
            dim=0)
        z_H = self.manifold_H.Frechet_mean(x_extend, keepdim=True, sum_idx=label_extend)
        return z_H

class CrossManifoldAttention(nn.Module):
    def __init__(self, manifold_q, manifold_k, in_dim, hidden_dim, out_dim, dropout):
        super(CrossManifoldAttention, self).__init__()
        self.manifold_q = manifold_q
        self.manifold_k = manifold_k
        self.q_lin = ConstCurveLinear(manifold_q, in_dim, hidden_dim, bias=False, dropout=dropout)
        self.k_lin = ConstCurveLinear(manifold_k, in_dim, hidden_dim, bias=False, dropout=dropout)
        self.v_lin = ConstCurveLinear(manifold_k, in_dim, hidden_dim, bias=False, dropout=dropout)
        self.scalar_map = nn.Sequential(
            nn.Linear(2 * hidden_dim, 1, bias=False),
            nn.LeakyReLU()
        )
        self.proj = ConstCurveLinear(manifold_k, hidden_dim, out_dim, bias=False, dropout=dropout)

    def forward(self, x_q, x_k, x_v, edge_index, agg_index=None):
        q = self.q_lin(x_q)
        k = self.k_lin(x_k)
        v = self.v_lin(x_v)
        src, dst = edge_index[0], edge_index[1]
        agg_index = agg_index if agg_index is not None else src
        
        # 🔧 筛选在有效范围内的索引
        num_nodes_q = q.size(0)
        num_nodes_k = k.size(0)
        num_nodes_v = v.size(0)
        
        # 检查索引有效性
        valid_mask = (src >= 0) & (src < num_nodes_q) & (dst >= 0) & (dst < num_nodes_k)
        
        if not valid_mask.all():
            # 过滤无效索引
            src = src[valid_mask]
            dst = dst[valid_mask]
            if agg_index is not None and len(agg_index) == len(valid_mask):
                agg_index = agg_index[valid_mask]
            else:
                agg_index = src
        
        # 如果没有有效边，返回零张量
        if len(src) == 0:
            return self.manifold_k.origin((num_nodes_q, self.proj.out_features), device=q.device)
        
        qk = torch.cat([q[src], k[dst]], dim=-1)
        score = self.scalar_map(qk).squeeze(-1)
        score = scatter_softmax(score, src, dim=-1)

        out = scatter_sum(score.unsqueeze(1) * v[dst], agg_index, dim=0, 
                         out=self.manifold_k.origin(q.shape, device=q.device))

        denorm = self.manifold_k.inner(None, out, keepdim=True)
        denorm = denorm.abs().clamp_min(1e-8).sqrt()
        out = 1. / self.manifold_k.k.sqrt() * out / denorm
        out = self.proj(out)
        return out