import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import numpy as np
from torch import Tensor
import logging
from torch_scatter import scatter_add

from tqdm import tqdm
from sampler_core import ParallelSampler
import torch_sparse


import time
import copy
from torch_sparse import SparseTensor
from torchmetrics.classification import MulticlassAUROC, MulticlassAveragePrecision
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from sklearn.preprocessing import MinMaxScaler
import os
import pickle


def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class PositionalEncoding1D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding1D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels
        inv_freq = 1.0 / (1000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = get_emb(sin_inp_x)
        emb = torch.zeros((x, self.channels), device=tensor.device).type(tensor.type())
        emb[:, : self.channels] = emb_x

        self.cached_penc = emb[None, :, :orig_ch].repeat(batch_size, 1, 1)
        return self.cached_penc


class PositionalEncodingPermute1D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x) instead of (batchsize, x, ch)
        """
        super(PositionalEncodingPermute1D, self).__init__()
        self.penc = PositionalEncoding1D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 2, 1)

    @property
    def org_channels(self):
        return self.penc.org_channels


class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 4) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1)
        emb_y = get_emb(sin_inp_y)
        emb = torch.zeros((x, y, self.channels * 2), device=tensor.device).type(
            tensor.type()
        )
        emb[:, :, : self.channels] = emb_x
        emb[:, :, self.channels : 2 * self.channels] = emb_y

        self.cached_penc = emb[None, :, :, :orig_ch].repeat(tensor.shape[0], 1, 1, 1)
        return self.cached_penc


class PositionalEncodingPermute2D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x, y) instead of (batchsize, x, y, ch)
        """
        super(PositionalEncodingPermute2D, self).__init__()
        self.penc = PositionalEncoding2D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 3, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 3, 1, 2)

    @property
    def org_channels(self):
        return self.penc.org_channels


class PositionalEncoding3D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding3D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, tensor):
        """
        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)
        """
        if len(tensor.shape) != 5:
            raise RuntimeError("The input tensor has to be 5d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, y, z, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        pos_z = torch.arange(z, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1).unsqueeze(1)
        emb_y = get_emb(sin_inp_y).unsqueeze(1)
        emb_z = get_emb(sin_inp_z)
        emb = torch.zeros((x, y, z, self.channels * 3), device=tensor.device).type(
            tensor.type()
        )
        emb[:, :, :, : self.channels] = emb_x
        emb[:, :, :, self.channels : 2 * self.channels] = emb_y
        emb[:, :, :, 2 * self.channels :] = emb_z

        self.cached_penc = emb[None, :, :, :, :orig_ch].repeat(batch_size, 1, 1, 1, 1)
        return self.cached_penc


class PositionalEncodingPermute3D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x, y, z) instead of (batchsize, x, y, z, ch)
        """
        super(PositionalEncodingPermute3D, self).__init__()
        self.penc = PositionalEncoding3D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 3, 4, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 4, 1, 2, 3)

    @property
    def org_channels(self):
        return self.penc.org_channels


class Summer(nn.Module):
    def __init__(self, penc):
        """
        :param model: The type of positional encoding to run the summer on.
        """
        super(Summer, self).__init__()
        self.penc = penc

    def forward(self, tensor):
        """
        :param tensor: A 3, 4 or 5d tensor that matches the model output size
        :return: Positional Encoding Matrix summed to the original tensor
        """
        penc = self.penc(tensor)
        assert (
            tensor.size() == penc.size()
        ), "The original tensor size {} and the positional encoding tensor size {} must match!".format(
            tensor.size(), penc.size()
        )
        return tensor + penc


"""
Source: STHN model.py
URL: https://github.com/celi52/STHN/blob/main/model.py
"""


"""
Module: Time-encoder
"""


class TimeEncode(nn.Module):
    """
    out = linear(time_scatter): 1-->time_dims
    out = cos(out)
    """

    def __init__(self, dim):
        super(TimeEncode, self).__init__()
        self.dim = dim
        self.w = nn.Linear(1, dim)
        self.reset_parameters()

    def reset_parameters(
        self,
    ):
        self.w.weight = nn.Parameter(
            (
                torch.from_numpy(
                    1 / 10 ** np.linspace(0, 9, self.dim, dtype=np.float32)
                )
            ).reshape(self.dim, -1)
        )
        self.w.bias = nn.Parameter(torch.zeros(self.dim))

        self.w.weight.requires_grad = False
        self.w.bias.requires_grad = False

    @torch.no_grad()
    def forward(self, t):
        output = torch.cos(self.w(t.reshape((-1, 1))))
        return output


################################################################################################
################################################################################################
################################################################################################
"""
Module: STHN
"""


class FeedForward(nn.Module):
    """
    2-layer MLP with GeLU (fancy version of ReLU) as activation
    """

    def __init__(self, dims, expansion_factor, dropout=0, use_single_layer=False):
        super().__init__()

        self.dims = dims
        self.use_single_layer = use_single_layer

        self.expansion_factor = expansion_factor
        self.dropout = dropout

        if use_single_layer:
            self.linear_0 = nn.Linear(dims, dims)
        else:
            self.linear_0 = nn.Linear(dims, int(expansion_factor * dims))
            self.linear_1 = nn.Linear(int(expansion_factor * dims), dims)

        self.reset_parameters()

    def reset_parameters(self):
        self.linear_0.reset_parameters()
        if self.use_single_layer == False:
            self.linear_1.reset_parameters()

    def forward(self, x):
        x = self.linear_0(x)
        x = F.gelu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.use_single_layer == False:
            x = self.linear_1(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class TransformerBlock(nn.Module):
    """
    out = X.T + MLP_Layernorm(X.T)     # apply token mixing
    out = out.T + MLP_Layernorm(out.T) # apply channel mixing
    """

    def __init__(
        self,
        dims,
        channel_expansion_factor=4,
        dropout=0.2,
        module_spec=None,
        use_single_layer=False,
    ):
        super().__init__()

        if module_spec == None:
            self.module_spec = ["token", "channel"]
        else:
            self.module_spec = module_spec.split("+")

        self.dims = dims
        if "token" in self.module_spec:
            self.transformer_encoder = _MultiheadAttention(
                d_model=dims, n_heads=2, d_k=None, d_v=None, attn_dropout=dropout
            )
        if "channel" in self.module_spec:
            self.channel_layernorm = nn.LayerNorm(dims)
            self.channel_forward = FeedForward(
                dims, channel_expansion_factor, dropout, use_single_layer
            )

    def reset_parameters(self):
        if "token" in self.module_spec:
            self.transformer_encoder.reset_parameters()
        if "channel" in self.module_spec:
            self.channel_layernorm.reset_parameters()
            self.channel_forward.reset_parameters()

    def token_mixer(self, x):
        x = self.transformer_encoder(x, x, x)
        return x

    def channel_mixer(self, x):
        x = self.channel_layernorm(x)
        x = self.channel_forward(x)
        return x

    def forward(self, x):
        if "token" in self.module_spec:
            x = x + self.token_mixer(x)
        if "channel" in self.module_spec:
            x = x + self.channel_mixer(x)
        return x


class _MultiheadAttention(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        d_k=None,
        d_v=None,
        res_attention=False,
        attn_dropout=0.0,
        proj_dropout=0.0,
        qkv_bias=True,
        lsa=False,
    ):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(
            d_model,
            n_heads,
            attn_dropout=attn_dropout,
            res_attention=self.res_attention,
            lsa=lsa,
        )

        # Poject output
        self.to_out = nn.Sequential(
            nn.Linear(n_heads * d_v, d_model), nn.Dropout(attn_dropout)
        )

    def reset_parameters(self):
        self.to_out[0].reset_parameters()
        self.W_Q.reset_parameters()
        self.W_K.reset_parameters()
        self.W_V.reset_parameters()

    def forward(
        self,
        Q: Tensor,
        K: Optional[Tensor] = None,
        V: Optional[Tensor] = None,
        prev: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ):

        bs = Q.size(0)
        if K is None:
            K = Q
        if V is None:
            V = Q

        # Linear (+ split in multiple heads)
        q_s = (
            self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        )  # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = (
            self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3, 1)
        )  # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = (
            self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1, 2)
        )  # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        output, attn_weights = self.sdp_attn(
            q_s,
            k_s,
            v_s,
            prev=prev,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
        )

        # back to the original inputs dimensions
        output = (
            output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v)
        )  # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        return output


class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(
        self, d_model, n_heads, attn_dropout=0.0, res_attention=False, lsa=False
    ):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim**-0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        prev: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ):
        """
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        """

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = (
            torch.matmul(q, k) * self.scale
        )  # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None:
            attn_scores = attn_scores + prev

        # Attention mask (optional)
        if (
            attn_mask is not None
        ):  # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if (
            key_padding_mask is not None
        ):  # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(
                key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf
            )

        # normalize the attention weights
        attn_weights = F.softmax(
            attn_scores, dim=-1
        )  # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(
            attn_weights, v
        )  # output: [bs x n_heads x max_q_len x d_v]

        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights


class FeatEncode(nn.Module):
    """
    Return [raw_edge_feat | TimeEncode(edge_time_stamp)]
    """

    def __init__(self, time_dims, feat_dims, out_dims):
        super().__init__()

        self.time_encoder = TimeEncode(time_dims)
        self.feat_encoder = nn.Linear(time_dims + feat_dims, out_dims)
        self.reset_parameters()

    def reset_parameters(self):
        self.time_encoder.reset_parameters()
        self.feat_encoder.reset_parameters()

    def forward(self, edge_feats, edge_ts):
        edge_time_feats = self.time_encoder(edge_ts)
        x = torch.cat([edge_feats, edge_time_feats], dim=1)
        return self.feat_encoder(x)


class Patch_Encoding(nn.Module):
    """
    Input : [ batch_size, graph_size, edge_dims+time_dims]
    Output: [ batch_size, graph_size, output_dims]
    """

    def __init__(
        self,
        per_graph_size,
        time_channels,
        input_channels,
        hidden_channels,
        out_channels,
        num_layers,
        dropout,
        channel_expansion_factor,
        window_size,
        module_spec=None,
        use_single_layer=False,
    ):
        super().__init__()
        self.per_graph_size = per_graph_size
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers

        # input & output classifer
        self.feat_encoder = FeatEncode(time_channels, input_channels, hidden_channels)
        self.layernorm = nn.LayerNorm(hidden_channels)
        self.mlp_head = nn.Linear(hidden_channels, out_channels)

        # inner layers
        self.mixer_blocks = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.mixer_blocks.append(
                TransformerBlock(
                    hidden_channels,
                    channel_expansion_factor,
                    dropout,
                    module_spec=None,
                    use_single_layer=use_single_layer,
                )
            )
        # padding
        self.stride = window_size
        self.window_size = window_size
        self.pad_projector = nn.Linear(window_size * hidden_channels, hidden_channels)
        self.p_enc_1d_model_sum = Summer(PositionalEncoding1D(hidden_channels))

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.mixer_blocks:
            layer.reset_parameters()
        self.feat_encoder.reset_parameters()
        self.layernorm.reset_parameters()
        self.mlp_head.reset_parameters()

    def forward(self, edge_feats, edge_ts, batch_size, inds):
        """
        前向传播方法，处理输入的边特征和时间戳，生成最终的特征表示。

        Args:
            edge_feats (torch.Tensor):  形状为[num_edges, edge_feature_dim]的边特征张量
            edge_ts (torch.Tensor): 形状为[num_edges]的时间差张量
            batch_size (int): 标量整数，表示子图数量，batch_size指的是原batch_szie*(源节点+目的节点+负采样节点)
            inds (torch.Tensor): 形状为[num_valid_edges]的索引张量

        Returns:
            torch.Tensor: 经过处理后的特征张量。生成所有节点的特征表示
        """
        # x : [ batch_size, graph_size, edge_dims+time_dims]
        # 使用特征编码器对边特征和时间戳进行编码
        edge_time_feats = self.feat_encoder(edge_feats, edge_ts)
        # 初始化一个全零张量，用于存储处理后的特征
        # 每per_graph_size为一个子图
        x = torch.zeros(
            (batch_size * self.per_graph_size, edge_time_feats.size(1)),
            device=edge_feats.device,
        )
        # 将编码后的边时间特征累加到对应索引位置
        x[inds] = x[inds] + edge_time_feats
        # 调整张量形状，将其分割为多个窗口
        x = x.view(
            -1, self.per_graph_size // self.window_size, self.window_size * x.shape[-1]
        )
        # 使用投影层对窗口特征进行投影
        x = self.pad_projector(x)
        # 添加一维位置编码
        x = self.p_enc_1d_model_sum(x)
        # 遍历所有的混合块，对特征进行处理
        for i in range(self.num_layers):
            # 对通道和特征维度应用混合块
            x = self.mixer_blocks[i](x)
        # 使用层归一化处理特征
        x = self.layernorm(x)
        # 对特征在维度1上求均值
        x = torch.mean(x, dim=1)
        # 使用全连接层生成最终的特征表示
        x = self.mlp_head(x)
        return x


################################################################################################
################################################################################################
################################################################################################

"""
Edge predictor
"""


class EdgePredictor_per_node(torch.nn.Module):
    """
    out = linear(src_node_feats) + linear(dst_node_feats)
    out = ReLU(out)
    """

    def __init__(self, dim_in_time, dim_in_node, predict_class):
        super().__init__()

        self.dim_in_time = dim_in_time
        self.dim_in_node = dim_in_node

        # dim_in_time + dim_in_node
        self.src_fc = torch.nn.Linear(dim_in_time + dim_in_node, 100)
        self.dst_fc = torch.nn.Linear(dim_in_time + dim_in_node, 100)

        self.out_fc = torch.nn.Linear(100, predict_class)
        self.reset_parameters()

    def reset_parameters(
        self,
    ):
        self.src_fc.reset_parameters()
        self.dst_fc.reset_parameters()
        self.out_fc.reset_parameters()

    def forward(self, h, neg_samples=1):
        num_edge = h.shape[0] // (neg_samples + 2)
        h_src = self.src_fc(h[:num_edge])
        h_pos_dst = self.dst_fc(h[num_edge : 2 * num_edge])
        h_neg_dst = self.dst_fc(h[2 * num_edge :])

        h_pos_edge = torch.nn.functional.relu(h_src + h_pos_dst)
        h_neg_edge = torch.nn.functional.relu(h_src.tile(neg_samples, 1) + h_neg_dst)

        return self.out_fc(h_pos_edge), self.out_fc(h_neg_edge)


# class STHN_Interface(nn.Module):
#     def __init__(self, mlp_mixer_configs, edge_predictor_configs):
#         super(STHN_Interface, self).__init__()

#         self.time_feats_dim = edge_predictor_configs["dim_in_time"]
#         self.node_feats_dim = edge_predictor_configs["dim_in_node"]

#         if self.time_feats_dim > 0:
#             self.base_model = Patch_Encoding(**mlp_mixer_configs)

#         self.edge_predictor = EdgePredictor_per_node(**edge_predictor_configs)
#         self.creterion = nn.CrossEntropyLoss(reduction="mean")
#         self.reset_parameters()

#     def reset_parameters(self):
#         if self.time_feats_dim > 0:
#             self.base_model.reset_parameters()
#         self.edge_predictor.reset_parameters()

#     def forward(self, model_inputs, neg_samples, node_feats):
#         pred_pos, pred_neg = self.predict(model_inputs, neg_samples, node_feats)
#         all_pred = torch.cat((pred_pos, pred_neg), dim=0)
#         all_edge_label = torch.cat(
#             (torch.ones_like(pred_pos), torch.zeros_like(pred_neg)), dim=0
#         )
#         loss = self.creterion(all_pred, all_edge_label).mean()
#         return loss, all_pred, all_edge_label

#     def predict(self, model_inputs, neg_samples, node_feats):
#         if self.time_feats_dim > 0 and self.node_feats_dim == 0:
#             x = self.base_model(*model_inputs)
#         elif self.time_feats_dim > 0 and self.node_feats_dim > 0:
#             x = self.base_model(*model_inputs)
#             x = torch.cat([x, node_feats], dim=1)
#         elif self.time_feats_dim == 0 and self.node_feats_dim > 0:
#             x = node_feats
#         else:
#             logging.info("Either time_feats_dim or node_feats_dim must larger than 0!")

#         pred_pos, pred_neg = self.edge_predictor(x, neg_samples=neg_samples)
#         return pred_pos, pred_neg


class STHN_Interface(nn.Module):
    def __init__(
        self,
        mlp_mixer_configs: dict,
        edge_predictor_configs: dict,
        llm_rgcn_configs: Optional[dict] = None,
    ):
        """
        [重大修改] __init__ 函数现在接收一个额外的 `llm_rgcn_configs` 字典。
        如果提供了这个字典，模型将以“增强模式”运行；否则，将以后向兼容的“原始模式”运行。

        `llm_rgcn_configs` 字典应包含:
            - 'dataset_name' (str): 用于定位LLM嵌入文件。
            - 'relation_emb_dim' (int): LLM关系嵌入的维度。
            - 'num_gnn_layers' (int): GNN层数。
        """
        super(STHN_Interface, self).__init__()

        # --- 原始模块初始化 ---
        self.time_feats_dim = edge_predictor_configs["dim_in_time"]
        self.node_feats_dim = edge_predictor_configs["dim_in_node"]
        if self.time_feats_dim > 0:
            self.base_model = Patch_Encoding(**mlp_mixer_configs)

        # --- [新增] 根据是否传入llm_rgcn_configs来决定运行模式 ---
        self.enhanced_mode = llm_rgcn_configs is not None

        if self.enhanced_mode:
            print("INFO: STHN_Interface 正在以 [LLM增强模式] 初始化...")
            temporal_out_dim = mlp_mixer_configs.get("out_channels")

            # Part 1: 加载LLM知识资产
            relation_emb_dim = llm_rgcn_configs["relation_emb_dim"]
            dataset_name = llm_rgcn_configs["dataset_name"]
            relation_emb_path = f'tgb/DATA/{dataset_name.replace("-", "_")}/relation_embs_emb{relation_emb_dim}.pt'
            if not os.path.exists(relation_emb_path):
                raise FileNotFoundError(
                    f"错误: 关系嵌入文件未找到! 请先运行第一步的脚本。\n路径: {relation_emb_path}"
                )

            relation_embs_list = torch.load(relation_emb_path, map_location="cpu")
            relation_embs_tensor = torch.stack(relation_embs_list, dim=0)
            self.register_buffer("relation_embs", relation_embs_tensor)

            # Part 2: 初始化GNN层
            num_gnn_layers = llm_rgcn_configs.get("num_gnn_layers", 2)
            self.gnn_layers = nn.ModuleList()
            self.gnn_layers.append(
                LLM_Enhanced_RGCNConv(
                    temporal_out_dim, temporal_out_dim, relation_emb_dim
                )
            )
            for _ in range(num_gnn_layers - 1):
                self.gnn_layers.append(
                    LLM_Enhanced_RGCNConv(
                        temporal_out_dim, temporal_out_dim, relation_emb_dim
                    )
                )

            # Part 3: 初始化融合层和修改后的预测器
            self.fusion_norm = nn.LayerNorm(temporal_out_dim)

            final_pred_configs = edge_predictor_configs.copy()
            final_pred_configs["dim_in_time"] = temporal_out_dim
            final_pred_configs["dim_in_node"] = 0  # 最终特征是融合后的单一向量
            self.edge_predictor = EdgePredictor_per_node(**final_pred_configs)

        else:
            print("INFO: STHN_Interface 正在以 [原始模式] 初始化...")
            # 在原始模式下，一切保持原样
            self.edge_predictor = EdgePredictor_per_node(**edge_predictor_configs)

        self.creterion = nn.CrossEntropyLoss(reduction="mean")
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, "base_model"):
            self.base_model.reset_parameters()
        if self.enhanced_mode:
            for layer in self.gnn_layers:
                layer.reset_parameters()
            self.fusion_norm.reset_parameters()
        self.edge_predictor.reset_parameters()

    def forward(
        self,
        model_inputs,
        neg_samples,
        node_feats,
        structural_inputs: Optional[Tuple] = None,
    ):
        """
        [修改] forward函数签名保持不变，但在末尾新增了一个可选参数 `structural_inputs`。
        """
        pred_pos, pred_neg = self.predict(
            model_inputs, neg_samples, node_feats, structural_inputs
        )

        # --- 输出格式与原始forward完全一致 ---
        all_pred = torch.cat((pred_pos, pred_neg), dim=0)
        # 假设是二分类链接预测
        pos_labels = torch.ones(pred_pos.shape[0], device=pred_pos.device)
        neg_labels = torch.zeros(pred_neg.shape[0], device=pred_neg.device)
        all_edge_label = torch.cat([pos_labels, neg_labels], dim=0)

        loss = self.creterion(all_pred.squeeze(), all_edge_label)
        return loss, all_pred, all_edge_label

    def predict(
        self,
        model_inputs,
        neg_samples,
        node_feats,
        structural_inputs: Optional[Tuple] = None,
    ):
        """
        [修改] predict函数同样新增了 `structural_inputs`。
        内部逻辑会根据是否为增强模式进行切换。
        """
        if self.enhanced_mode:
            if structural_inputs is None:
                raise ValueError(
                    "增强模式需要提供 structural_inputs (edge_index, edge_type)!"
                )

            # --- 新的增强数据流 ---
            # 1. 时序编码
            h_temporal = self.base_model(*model_inputs)
            # 2. GNN增强
            edge_index, edge_type = structural_inputs
            h_structural = h_temporal
            for layer in self.gnn_layers:
                h_structural = layer(
                    h_structural, edge_index, edge_type, self.relation_embs
                )
            # 3. 融合
            x = self.fusion_norm(h_temporal + h_structural)

        else:
            # --- 保持原始数据流不变 ---
            if self.time_feats_dim > 0 and self.node_feats_dim == 0:
                x = self.base_model(*model_inputs)
            elif self.time_feats_dim > 0 and self.node_feats_dim > 0:
                x = self.base_model(*model_inputs)
                x = torch.cat([x, node_feats], dim=1)
            elif self.time_feats_dim == 0 and self.node_feats_dim > 0:
                x = node_feats
            else:
                raise ValueError("原始模式下，time_feats_dim或node_feats_dim必须大于0!")

        # 4. 最终预测 (两种模式共用)
        pred_pos, pred_neg = self.edge_predictor(x, neg_samples=neg_samples)
        return pred_pos, pred_neg


class Multiclass_Interface(nn.Module):
    def __init__(self, mlp_mixer_configs, edge_predictor_configs):
        super(Multiclass_Interface, self).__init__()

        self.time_feats_dim = edge_predictor_configs["dim_in_time"]
        self.node_feats_dim = edge_predictor_configs["dim_in_node"]

        if self.time_feats_dim > 0:
            self.base_model = Patch_Encoding(**mlp_mixer_configs)

        self.edge_predictor = EdgePredictor_per_node(**edge_predictor_configs)
        self.creterion = nn.CrossEntropyLoss(reduction="mean")
        self.reset_parameters()

    def reset_parameters(self):
        if self.time_feats_dim > 0:
            self.base_model.reset_parameters()
        self.edge_predictor.reset_parameters()

    def forward(self, model_inputs, neg_samples, node_feats):
        pos_edge_label = model_inputs[-1].view(-1, 1)
        model_inputs = model_inputs[:-1]
        pred_pos, pred_neg = self.predict(model_inputs, neg_samples, node_feats)

        all_pred = torch.cat((pred_pos, pred_neg), dim=0)
        all_edge_label = torch.squeeze(
            torch.cat((pos_edge_label, torch.zeros_like(pos_edge_label)), dim=0)
        )
        loss = self.creterion(all_pred, all_edge_label).mean()

        return loss, all_pred, all_edge_label

    def predict(self, model_inputs, neg_samples, node_feats):
        if self.time_feats_dim > 0 and self.node_feats_dim == 0:
            x = self.base_model(*model_inputs)
        elif self.time_feats_dim > 0 and self.node_feats_dim > 0:
            x = self.base_model(*model_inputs)
            x = torch.cat([x, node_feats], dim=1)
        elif self.time_feats_dim == 0 and self.node_feats_dim > 0:
            x = node_feats
        else:
            logging.info("Either time_feats_dim or node_feats_dim must larger than 0!")

        pred_pos, pred_neg = self.edge_predictor(x, neg_samples=neg_samples)
        return pred_pos, pred_neg


class LLM_Enhanced_RGCNConv(nn.Module):
    """
    一个由LLM增强的关系图卷积网络层 (LLM-Enhanced Relational Graph Convolutional Network Layer)。

    它在进行消息传递时，会将源节点的特征与连接边的“LLM关系嵌入”进行拼接，
    从而使得传递的消息同时包含结构信息和丰富的语义信息。
    """

    def __init__(self, in_channels: int, out_channels: int, relation_emb_dim: int):
        """
        初始化GNN层。

        Args:
            in_channels (int): 输入节点特征的维度。
            out_channels (int): 输出节点特征的维度。
            relation_emb_dim (int): LLM生成的关系嵌入的维度 (例如 1536)。
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 核心：一个MLP，用于处理拼接后的向量 [节点特征, LLM关系嵌入]
        # 它的输入维度是 in_channels + relation_emb_dim
        self.message_mlp = nn.Sequential(
            nn.Linear(in_channels + relation_emb_dim, in_channels * 2),
            nn.ReLU(),
            nn.Linear(in_channels * 2, out_channels),
        )

        # 节点自身信息更新的线性层（用于GNN中的自环 self-loop）
        self.self_loop_mlp = nn.Linear(in_channels, out_channels)

        # 添加归一化层以稳定训练
        self.norm = nn.LayerNorm(out_channels)
        self.activation = nn.ReLU()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        relation_embs_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        执行一次前向传播（消息传递和聚合）。

        Args:
            x (torch.Tensor): 节点特征张量, shape [num_nodes, in_channels]
            edge_index (torch.Tensor): 图的边索引 (COO格式), shape [2, num_edges]
            edge_type (torch.Tensor): 每条边的类型ID, shape [num_edges]
            relation_embs_tensor (torch.Tensor): 所有关系类型的LLM嵌入, shape [num_relation_types, relation_emb_dim]

        Returns:
            torch.Tensor: 更新后的节点特征张量, shape [num_nodes, out_channels]
        """
        # 从边索引中获取源节点和目标节点
        source_nodes, dest_nodes = edge_index

        # 1. 根据edge_type(边的类型ID)从我们准备的资产中查找每条边对应的关系嵌入
        # shape: [num_edges, relation_emb_dim]
        edge_relation_embs = relation_embs_tensor[edge_type]

        # 2. 获取构成每条边的“源节点”的特征
        # shape: [num_edges, in_channels]
        source_node_feats = x[source_nodes]

        # 3. 【核心】将源节点特征和关系嵌入进行拼接。
        # 这一步就是奇迹发生的地方。我们将节点的纯粹的动态特征与边的纯粹的语义特征（来自LLM）结合在了一起。
        # shape: [num_edges, in_channels + relation_emb_dim]
        message_inputs = torch.cat([source_node_feats, edge_relation_embs], dim=-1)

        # 4. 将拼接后的向量通过MLP，生成最终要传递的信息 (message)
        # shape: [num_edges, out_channels]
        messages = self.message_mlp(message_inputs)

        # 5. 使用 scatter_add 高效地将所有发往同一目标节点的信息相加
        # shape: [num_nodes, out_channels]
        aggregated_messages = scatter_add(
            messages, dest_nodes, dim=0, dim_size=x.size(0)
        )

        # 6. 更新节点表示：聚合的邻居信息 + 自身信息变换
        # W_0 * h_i + sum(W_r * h_j)
        out = self.self_loop_mlp(x) + aggregated_messages

        # 7. 应用归一化和激活函数，得到最终输出
        out = self.activation(self.norm(out))

        return out
