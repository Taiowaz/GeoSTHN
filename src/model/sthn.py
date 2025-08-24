from turtle import pos
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import numpy as np
from torch import Tensor
import logging


from tqdm import tqdm
from sampler_core import ParallelSampler
import torch_sparse


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


class STHN_Interface(nn.Module):
    def __init__(self, mlp_mixer_configs, edge_predictor_configs):
        super(STHN_Interface, self).__init__()

        self.time_feats_dim = edge_predictor_configs["dim_in_time"]
        self.node_feats_dim = edge_predictor_configs["dim_in_node"]

        if self.time_feats_dim > 0:
            self.base_model = Patch_Encoding(**mlp_mixer_configs)

        self.edge_predictor = EdgePredictor_per_node(**edge_predictor_configs)
        # 二分类损失函数
        self.criterion = nn.BCEWithLogitsLoss(reduction="mean")
        self.reset_parameters()

    def reset_parameters(self):
        if self.time_feats_dim > 0:
            self.base_model.reset_parameters()
        self.edge_predictor.reset_parameters()

    def forward(self, model_inputs, neg_samples, node_feats):
        pred_pos, pred_neg = self.predict(model_inputs, neg_samples, node_feats)
        all_pred = torch.cat((pred_pos, pred_neg), dim=0)
        all_edge_label = torch.cat(
            (torch.ones_like(pred_pos), torch.zeros_like(pred_neg)), dim=0
        )
        loss = self.criterion(all_pred, all_edge_label).mean()
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


class Multiclass_Interface(nn.Module):
    def __init__(self, mlp_mixer_configs, edge_predictor_configs):
        super(Multiclass_Interface, self).__init__()

        self.time_feats_dim = edge_predictor_configs["dim_in_time"]
        self.node_feats_dim = edge_predictor_configs["dim_in_node"]

        if self.time_feats_dim > 0:
            self.base_model = Patch_Encoding(**mlp_mixer_configs)

        self.edge_predictor = EdgePredictor_per_node(**edge_predictor_configs)
        self.criterion = nn.CrossEntropyLoss(reduction="mean")
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
        loss = self.criterion(all_pred, all_edge_label).mean()

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






class HeteroTimeEncode(nn.Module):
    """
    异构时间编码器 - 为不同类型的边提供专门的时间编码
    基于原有TimeEncode扩展，保持接口兼容性
    """
    def __init__(self, edge_types: list, time_dim: int = 100):
        super(HeteroTimeEncode, self).__init__()
        self.edge_types = edge_types
        self.time_dim = time_dim
        self.num_edge_types = len(edge_types)
        
        # 为每种边类型创建专门的时间编码器
        self.type_encoders = nn.ModuleDict()
        for i, edge_type in enumerate(edge_types):
            encoder = TimeEncode(time_dim)
            # 为不同类型设置不同的频率分布，避免重叠
            # 通过调整权重来实现频率偏移
            freq_multiplier = 1.0 + i * 0.1  # 每种类型有10%的频率偏移
            encoder.w.weight.data *= freq_multiplier
            self.type_encoders[str(edge_type)] = encoder
        
        # 默认编码器（用于兼容性）
        self.default_encoder = TimeEncode(time_dim)
        
    def forward(self, edge_ts: torch.Tensor, edge_types: torch.Tensor = None):
        """
        前向传播
        
        Args:
            edge_ts: [num_edges] 时间戳张量
            edge_types: [num_edges] 边类型索引张量，可选
                       如果为None，则所有边使用默认编码器
        
        Returns:
            torch.Tensor: [num_edges, time_dim] 时间特征嵌入
        """
        if edge_types is None:
            # 如果没有类型信息，使用默认编码器（向后兼容）
            return self.default_encoder(edge_ts)
        
        # 初始化输出张量
        batch_size = edge_ts.shape[0]
        time_embeddings = torch.zeros(batch_size, self.time_dim, 
                                    device=edge_ts.device, dtype=edge_ts.dtype)
        
        # 为每种边类型分别编码
        for i, edge_type in enumerate(self.edge_types):
            # 找到当前类型的边
            type_mask = (edge_types == i)
            if type_mask.any():
                # 获取当前类型的时间戳
                type_times = edge_ts[type_mask]
                # 使用对应的编码器
                type_encoder = self.type_encoders[str(edge_type)]
                type_embeddings = type_encoder(type_times)
                # 存储到对应位置
                time_embeddings[type_mask] = type_embeddings
        
        return time_embeddings
    
    def reset_parameters(self):
        """重置所有编码器的参数"""
        for encoder in self.type_encoders.values():
            encoder.reset_parameters()
        self.default_encoder.reset_parameters()


class HeteroFeatEncode(nn.Module):
    """
    异构特征编码器 - 为不同类型的边提供专门的特征编码
    基于原有FeatEncode扩展，保持接口兼容性
    Return [raw_edge_feat | HeteroTimeEncode(edge_time_stamp)] + type_embedding
    """
    def __init__(self, edge_types: list, time_dims: int, feat_dims: int, out_dims: int):
        super(HeteroFeatEncode, self).__init__()
        self.edge_types = edge_types
        self.time_dims = time_dims
        self.feat_dims = feat_dims
        self.out_dims = out_dims
        
        # 🆕 NEW: 使用异构时间编码器替代原有的单一时间编码器
        self.time_encoder = HeteroTimeEncode(edge_types, time_dims)
        
        # 🆕 NEW: 为每种边类型创建专门的特征编码器（原来只有一个）
        self.feat_encoders = nn.ModuleDict()
        for edge_type in edge_types:
            self.feat_encoders[str(edge_type)] = nn.Linear(time_dims + feat_dims, out_dims)
        
        # 🆕 NEW: 添加类型嵌入层（原来没有）
        self.edge_type_embedding = nn.Embedding(len(edge_types), out_dims)
        
        # 🆕 NEW: 默认特征编码器（用于向后兼容，原来的FeatEncode逻辑）
        self.default_feat_encoder = nn.Linear(time_dims + feat_dims, out_dims)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.time_encoder.reset_parameters()
        # 🆕 NEW: 重置所有类型专门的编码器
        for encoder in self.feat_encoders.values():
            encoder.reset_parameters()
        # 🆕 NEW: 重置类型嵌入
        self.edge_type_embedding.reset_parameters()
        # 保持兼容性
        self.default_feat_encoder.reset_parameters()
    
    def forward(self, edge_feats: torch.Tensor, edge_ts: torch.Tensor, edge_types: torch.Tensor = None):
        """
        前向传播 - 保持与原有FeatEncode相同的接口
        
        Args:
            edge_feats: [num_edges, feat_dim] 边特征
            edge_ts: [num_edges] 时间戳
            edge_types: [num_edges] 边类型索引（🆕 NEW: 新增参数）
        
        Returns:
            torch.Tensor: [num_edges, out_dims] 编码后的特征
        """
        # 🆕 NEW: 使用异构时间编码器（原来用普通TimeEncode）
        edge_time_feats = self.time_encoder(edge_ts, edge_types)
        
        # 拼接边特征和时间特征（与原来相同）
        combined_feats = torch.cat([edge_feats, edge_time_feats], dim=1)
        
        if edge_types is None:
            # 🆕 NEW: 向后兼容模式 - 如果没有类型信息，使用默认编码器
            return self.default_feat_encoder(combined_feats)
        
        # 🆕 NEW: 异构模式 - 根据边类型分别编码（原来没有这个逻辑）
        output_feats = torch.zeros(len(edge_feats), self.out_dims, device=edge_feats.device)
        
        for i, edge_type in enumerate(self.edge_types):
            type_mask = (edge_types == i)
            if type_mask.any():
                # 使用对应类型的特征编码器
                type_feats = combined_feats[type_mask]
                type_output = self.feat_encoders[str(edge_type)](type_feats)
                
                # 🆕 NEW: 添加类型嵌入（原来没有）
                type_emb = self.edge_type_embedding(torch.tensor(i, device=edge_feats.device))
                type_output = type_output + type_emb.unsqueeze(0).expand(type_output.size(0), -1)
                
                output_feats[type_mask] = type_output
        
        return output_feats


class HeteroPatch_Encoding(nn.Module):
    """
    异构图的Patch编码器 - 保持与原有Patch_Encoding相同的接口
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
        edge_types: list = None,  # 🆕 NEW: 新增边类型参数（原来没有）
        module_spec=None,
        use_single_layer=False,
    ):
        super().__init__()
        self.per_graph_size = per_graph_size
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.edge_types = edge_types or ['0']  # 🆕 NEW: 默认单一类型（保持兼容性）
        
        # 🆕 NEW: 使用异构特征编码器替代原有的FeatEncode
        self.feat_encoder = HeteroFeatEncode(
            self.edge_types, time_channels, input_channels, hidden_channels
        )
        
        # 以下部分与原有Patch_Encoding完全相同
        self.layernorm = nn.LayerNorm(hidden_channels)
        self.mlp_head = nn.Linear(hidden_channels, out_channels)
        
        # inner layers - 保持原有的TransformerBlock结构
        self.mixer_blocks = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.mixer_blocks.append(
                TransformerBlock(
                    hidden_channels,
                    channel_expansion_factor,
                    dropout,
                    module_spec=module_spec,  # 🆕 NEW: 传递module_spec参数（原来写死为None）
                    use_single_layer=use_single_layer,
                )
            )
        
        # padding - 与原有逻辑完全相同
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
        self.pad_projector.reset_parameters()  # 🆕 NEW: 添加了这个重置（原来可能遗漏了）
    
    def forward(self, edge_feats, edge_ts, batch_size, inds, edge_types=None):  # 🆕 NEW: 新增edge_types参数
        """
        前向传播方法，处理输入的边特征和时间戳，生成最终的特征表示。
        保持与原有Patch_Encoding完全相同的接口

        Args:
            edge_feats (torch.Tensor):  形状为[num_edges, edge_feature_dim]的边特征张量
            edge_ts (torch.Tensor): 形状为[num_edges]的时间差张量
            batch_size (int): 标量整数，表示子图数量，batch_size指的是原batch_szie*(源节点+目的节点+负采样节点)
            inds (torch.Tensor): 形状为[num_valid_edges]的索引张量
            edge_types (torch.Tensor, optional): 🆕 NEW: 形状为[num_edges]的边类型张量

        Returns:
            torch.Tensor: 经过处理后的特征张量。生成所有节点的特征表示
        """
        # 🆕 NEW: 使用异构特征编码器（原来用普通FeatEncode）
        edge_time_feats = self.feat_encoder(edge_feats, edge_ts, edge_types)
        
        # 以下处理流程与原有Patch_Encoding完全一致
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


class HeteroEdgePredictor_per_node(torch.nn.Module):
    """
    异构边预测器 - 为不同类型的边提供专门的预测器
    基于原有EdgePredictor_per_node扩展，保持接口兼容性
    out = linear(src_node_feats) + linear(dst_node_feats)
    out = ReLU(out)
    """
    def __init__(self, dim_in_time, dim_in_node, predict_class, edge_types: list = None):
        super().__init__()
        
        self.dim_in_time = dim_in_time
        self.dim_in_node = dim_in_node
        self.predict_class = predict_class
        self.edge_types = edge_types or ['0']  # 🆕 NEW: 默认单一类型（保持兼容性）
        
        # 🆕 NEW: 为每种边类型创建专门的预测器（原来只有一组）
        self.predictors = nn.ModuleDict()
        for edge_type in self.edge_types:
            self.predictors[str(edge_type)] = nn.ModuleDict({
                'src_fc': torch.nn.Linear(dim_in_time + dim_in_node, 100),
                'dst_fc': torch.nn.Linear(dim_in_time + dim_in_node, 100),
                'out_fc': torch.nn.Linear(100, predict_class)
            })
        
        # 🆕 NEW: 默认预测器（用于向后兼容，原来的EdgePredictor_per_node逻辑）
        self.default_src_fc = torch.nn.Linear(dim_in_time + dim_in_node, 100)
        self.default_dst_fc = torch.nn.Linear(dim_in_time + dim_in_node, 100)
        self.default_out_fc = torch.nn.Linear(100, predict_class)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        # 🆕 NEW: 重置所有类型专门的预测器
        for predictor_dict in self.predictors.values():
            for layer in predictor_dict.values():
                layer.reset_parameters()
        
        # 保持兼容性
        self.default_src_fc.reset_parameters()
        self.default_dst_fc.reset_parameters()
        self.default_out_fc.reset_parameters()
    
    def forward(self, h, neg_samples=1, poss_edgetypes=None):  # 🆕 NEW: 新增edge_types参数
        """
        前向传播 - 保持与原有EdgePredictor_per_node相同的接口
        
        Args:
            h: [batch_size, feature_dim] 节点特征
            neg_samples: 负采样数量
            edge_types: [num_edges] 边类型索引（🆕 NEW: 新增参数）
        
        Returns:
            tuple: (正边预测结果, 负边预测结果)
        """
        num_edge = h.shape[0] // (neg_samples + 2)
        h_src = h[:num_edge]
        h_pos_dst = h[num_edge : 2 * num_edge]
        h_neg_dst = h[2 * num_edge :]
        
        if poss_edgetypes is None or len(self.edge_types) == 1:
            # 🆕 NEW: 向后兼容模式 - 如果没有类型信息或只有一种类型，使用默认预测器
            if len(self.edge_types) == 1:
                # 使用第一个（也是唯一的）类型预测器
                predictor = self.predictors[self.edge_types[0]]
                h_src_enc = predictor['src_fc'](h_src)
                h_pos_dst_enc = predictor['dst_fc'](h_pos_dst)
                h_neg_dst_enc = predictor['dst_fc'](h_neg_dst)
                
                h_pos_edge = torch.nn.functional.relu(h_src_enc + h_pos_dst_enc)
                h_neg_edge = torch.nn.functional.relu(h_src_enc.tile(neg_samples, 1) + h_neg_dst_enc)
                
                return predictor['out_fc'](h_pos_edge), predictor['out_fc'](h_neg_edge)
            else:
                # 使用默认预测器（完全兼容原来的逻辑）
                h_src_enc = self.default_src_fc(h_src)
                h_pos_dst_enc = self.default_dst_fc(h_pos_dst)
                h_neg_dst_enc = self.default_dst_fc(h_neg_dst)
                
                h_pos_edge = torch.nn.functional.relu(h_src_enc + h_pos_dst_enc)
                h_neg_edge = torch.nn.functional.relu(h_src_enc.tile(neg_samples, 1) + h_neg_dst_enc)
                
                return self.default_out_fc(h_pos_edge), self.default_out_fc(h_neg_edge)
        
        else:
            # 🆕 NEW: 异构模式 - 根据边类型分别预测（原来没有这个逻辑）
            return self._hetero_forward(h_src, h_pos_dst, h_neg_dst, poss_edgetypes, neg_samples)
    
    def _hetero_forward(self, h_src, h_pos_dst, h_neg_dst, poss_edgetypes, neg_samples):
        """
        🆕 NEW: 异构边预测的具体实现 - 根据edge_types掩码只使用可能的预测器，取最大分数
        
        Args:
            h_src: [num_edges, feature_dim] 源节点特征
            h_pos_dst: [num_edges, feature_dim] 正样本目标节点特征  
            h_neg_dst: [neg_samples * num_edges, feature_dim] 负样本目标节点特征
            poss_edgetypes: [(1 + neg_samples) * num_edges, num_edge_types] 边类型掩码，1表示可能存在该类型
            neg_samples: 负采样数量
        """
        num_src = h_src.shape[0]
        device = h_src.device
        
        # 初始化输出张量，使用极小值填充（-inf会在softmax时出问题）
        all_pos_preds = []  # 存储每个边的有效预测结果
        
        all_neg_preds = [[] for _ in range(neg_samples)] 
        
        # 为每条边分别处理
        for src_idx in range(num_src):

            # 对正样本进行处理
            pos_edgetypes_mask = poss_edgetypes[src_idx]  # [num_edge_types]
            pos_valid_types = torch.nonzero(pos_edgetypes_mask, as_tuple=False).squeeze(-1)  # 获取有效类型索引
            
            # !!!!!，是否存在问题
            if len(pos_valid_types) == 0:
                # 如果没有有效类型，使用零向量或默认预测
                pos_pred = torch.zeros(self.predict_class, device=device)
            else:
                # 当前边的特征
                curr_h_src = h_src[src_idx:src_idx+1]  # [1, feature_dim]
                curr_h_pos_dst = h_pos_dst[src_idx:src_idx+1]  # [1, feature_dim]
             
                # 存储当前边所有有效类型的预测结果
                edge_pos_preds = []  # [num_valid_types, predict_class]
  
                # 正样本预测
                for type_idx in pos_valid_types:
                    edge_type = self.edge_types[type_idx]
                    predictor = self.predictors[str(edge_type)]
                    
                    # 使用当前类型的预测器
                    type_h_src_enc = predictor['src_fc'](curr_h_src)
                    type_h_pos_dst_enc = predictor['dst_fc'](curr_h_pos_dst)
                    
                    # 计算边表示
                    type_h_pos_edge = torch.nn.functional.relu(type_h_src_enc + type_h_pos_dst_enc)

                    # 预测
                    type_pos_pred = predictor['out_fc'](type_h_pos_edge)  # [1, 1]
                    edge_pos_preds.append(type_pos_pred.squeeze(0))  # [num_pos_valid_types]
                
                # 将有效类型的预测结果堆叠并取最大值
                if len(edge_pos_preds) > 1:
                    # 多个有效类型，堆叠后取最大值
                    edge_pos_stack = torch.stack(edge_pos_preds, dim=0)  # [num_valid_types, predict_class] 
                    pos_pred, _ = torch.max(edge_pos_stack, dim=0)  # [predict_class]
                else:
                    # 只有一个有效类型
                    pos_pred = edge_pos_preds[0]  # [predict_class]
            all_pos_preds.append(pos_pred)

            # 负样本预测
            
            for neg_idx in range(neg_samples):
                global_neg_idx = src_idx + neg_idx * num_src
                neg_edgetypes_mask = poss_edgetypes[num_src + global_neg_idx]
                neg_valid_types = torch.nonzero(neg_edgetypes_mask, as_tuple=False).squeeze(-1)

                if len(neg_valid_types) == 0:
                    # 如果没有有效类型，使用零向量！！！或者随机？
                    neg_pred = torch.zeros(self.predict_class, device=device)
                else:
                    # 当前负样本的特征
                    curr_h_src = h_src[src_idx:src_idx+1]  # [1, feature_dim] 复用源节点
                    curr_h_neg_dst = h_neg_dst[global_neg_idx:global_neg_idx+1]  
                    edge_neg_preds = []
                    for type_idx in neg_valid_types:
                        edge_type = self.edge_types[type_idx]
                        predictor = self.predictors[str(edge_type)]

                        # 使用当前类型的预测器
                        type_h_src_enc = predictor['src_fc'](curr_h_src)
                        type_h_neg_dst_enc = predictor['dst_fc'](curr_h_neg_dst)

                        # 计算边表示
                        type_h_neg_edge = torch.nn.functional.relu(type_h_src_enc + type_h_neg_dst_enc)
                        # 预测
                        type_neg_pred = predictor['out_fc'](type_h_neg_edge)  # [1, 1]
                        edge_neg_preds.append(type_neg_pred.squeeze(0))  # [num_neg_valid_types]

                    # 将有效类型的预测结果堆叠并取最大值
                    if len(edge_neg_preds) > 1:
                        # 多个有效类型，堆叠后取最大值
                        edge_neg_stack = torch.stack(edge_neg_preds, dim=0)  # [num_valid_types, 1] 
                        neg_pred, _ = torch.max(edge_neg_stack, dim=0)  # [1]
                    else:
                        # 只有一个有效类型
                        neg_pred = edge_neg_preds[0]  # [1]
                all_neg_preds[neg_idx].append(neg_pred)


        # 将所有边的结果堆叠
        final_pos_pred = torch.stack(all_pos_preds, dim=0)  # [num_edges, 1]
        final_neg_pred_list = []
        for neg_idx in range(neg_samples):
            # 堆叠当前批次的所有负样本
            batch_neg_preds = torch.stack(all_neg_preds[neg_idx], dim=0)  # [num_edges, predict_class]
            final_neg_pred_list.append(batch_neg_preds)
        
        # 拼接所有批次
        final_neg_pred = torch.cat(final_neg_pred_list, dim=0)  # [num_edges * neg_samples, predict_class]
    
        return final_pos_pred, final_neg_pred

class HeteroSTHN_Interface(nn.Module):
    """
    异构STHN接口 - 保持与原有STHN_Interface完全相同的外部接口
    整合所有异构组件：HeteroPatch_Encoding + HeteroEdgePredictor_per_node
    """
    def __init__(self, mlp_mixer_configs, edge_predictor_configs, edge_types: list = None):
        super(HeteroSTHN_Interface, self).__init__()

        self.time_feats_dim = edge_predictor_configs["dim_in_time"]
        self.node_feats_dim = edge_predictor_configs["dim_in_node"]
        self.edge_types = edge_types or ['0']  # 🆕 NEW: 支持边类型（原来没有）

        # 🆕 NEW: 使用异构组件替代原有组件
        if self.time_feats_dim > 0:
            # 传递边类型信息给mlp_mixer_configs
            mlp_mixer_configs['edge_types'] = self.edge_types  # 🆕 NEW: 添加边类型配置
            self.base_model = HeteroPatch_Encoding(**mlp_mixer_configs)  # 🆕 NEW: 使用异构Patch编码器
        
        # 传递边类型信息给edge_predictor_configs
        edge_predictor_configs['edge_types'] = self.edge_types  # 🆕 NEW: 添加边类型配置
        self.edge_predictor = HeteroEdgePredictor_per_node(**edge_predictor_configs)  # 🆕 NEW: 使用异构边预测器
        
        # 损失函数保持不变
        self.criterion = nn.BCEWithLogitsLoss(reduction="mean")
        self.reset_parameters()

    def reset_parameters(self):
        if self.time_feats_dim > 0:
            self.base_model.reset_parameters()
        self.edge_predictor.reset_parameters()

    def forward(self, model_inputs, neg_samples, node_feats, poss_edgetypes): 
        """
        前向传播 - 保持与原有STHN_Interface相同的接口（只是新增了可选的edge_types参数）
        
        Args:
            model_inputs: 模型输入（边特征、时间戳、batch_size、索引）
            neg_samples: 负采样数量
            node_feats: 节点特征
            edge_types: [num_edges] 边类型索引（🆕 NEW: 新增参数，可选）
        
        Returns:
            tuple: (loss, all_pred, all_edge_label) - 与原来完全相同的输出格式
        """
        edge_feats = model_inputs[0]
        # edge_feats是边类型的onehot编码，需要转回边类型数组
        his_edgetypes = torch.argmax(edge_feats, dim=1)
        pred_pos, pred_neg = self.predict(model_inputs, neg_samples, node_feats, his_edgetypes,poss_edgetypes)
        
        # 损失计算逻辑与原来完全相同
        all_pred = torch.cat((pred_pos, pred_neg), dim=0)
        all_edge_label = torch.cat(
            (torch.ones_like(pred_pos), torch.zeros_like(pred_neg)), dim=0
        )
        loss = self.criterion(all_pred, all_edge_label).mean()
        
        return loss, all_pred, all_edge_label

    def predict(self, model_inputs, neg_samples, node_feats, his_edgetypes, poss_edgetypes):  # 🆕 NEW: 新增edge_types参数
        """
        预测方法 - 保持与原有STHN_Interface相同的逻辑，但支持边类型
        
        Args:
            model_inputs: 模型输入
            neg_samples: 负采样数量  
            node_feats: 节点特征
            edge_types: 边类型（🆕 NEW: 新增参数，可选）
        
        Returns:
            tuple: (正边预测, 负边预测)
        """
        # 🆕 NEW: 检查model_inputs是否包含边类型信息
        if len(model_inputs) == 5:
            # 如果model_inputs包含5个元素，最后一个是边类型
            edge_feats, edge_ts, batch_size, inds, input_edge_types = model_inputs
            # 优先使用传入的edge_types，如果没有则使用model_inputs中的
            edge_types = input_edge_types if edge_types is None else edge_types
            # 重新构造model_inputs为4元素版本（兼容原有接口）
            model_inputs_for_base = [edge_feats, edge_ts, batch_size, inds]
        else:
            # 原有的4元素版本
            model_inputs_for_base = model_inputs
        
        # 特征提取逻辑与原来相同，但传递边类型信息
        if self.time_feats_dim > 0 and self.node_feats_dim == 0:
            # 🆕 NEW: 向异构Patch编码器传递边类型信息
            x = self.base_model(*model_inputs_for_base, his_edgetypes)
        elif self.time_feats_dim > 0 and self.node_feats_dim > 0:
            # 🆕 NEW: 向异构Patch编码器传递边类型信息
            x = self.base_model(*model_inputs_for_base, his_edgetypes)
            x = torch.cat([x, node_feats], dim=1)
        elif self.time_feats_dim == 0 and self.node_feats_dim > 0:
            x = node_feats
        else:
            logging.info("Either time_feats_dim or node_feats_dim must larger than 0!")

        # 🆕 NEW: 向异构边预测器传递边类型信息
        pred_pos, pred_neg = self.edge_predictor(x, neg_samples=neg_samples, poss_edgetypes=poss_edgetypes)
        return pred_pos, pred_neg


class HeteroMulticlass_Interface(nn.Module):
    """
    异构多分类接口 - 基于原有Multiclass_Interface扩展
    """
    def __init__(self, mlp_mixer_configs, edge_predictor_configs, edge_types: list = None):
        super(HeteroMulticlass_Interface, self).__init__()

        self.time_feats_dim = edge_predictor_configs["dim_in_time"]
        self.node_feats_dim = edge_predictor_configs["dim_in_node"]
        self.edge_types = edge_types or ['0']  # 🆕 NEW: 支持边类型

        # 🆕 NEW: 使用异构组件
        if self.time_feats_dim > 0:
            mlp_mixer_configs['edge_types'] = self.edge_types
            self.base_model = HeteroPatch_Encoding(**mlp_mixer_configs)

        edge_predictor_configs['edge_types'] = self.edge_types
        self.edge_predictor = HeteroEdgePredictor_per_node(**edge_predictor_configs)
        
        # 多分类损失函数
        self.criterion = nn.CrossEntropyLoss(reduction="mean")
        self.reset_parameters()

    def reset_parameters(self):
        if self.time_feats_dim > 0:
            self.base_model.reset_parameters()
        self.edge_predictor.reset_parameters()

    def forward(self, model_inputs, neg_samples, node_feats, edge_types=None):  # 🆕 NEW: 新增edge_types参数
        """
        前向传播 - 保持与原有Multiclass_Interface相同的接口
        """
        # 🆕 NEW: 处理包含边类型的model_inputs
        if len(model_inputs) == 6:  # [edge_feats, edge_ts, batch_size, inds, pos_edge_label, edge_types]
            pos_edge_label = model_inputs[-2].view(-1, 1)
            edge_types = model_inputs[-1] if edge_types is None else edge_types
            model_inputs_for_predict = model_inputs[:-2]
        else:  # 原有格式 [edge_feats, edge_ts, batch_size, inds, pos_edge_label]
            pos_edge_label = model_inputs[-1].view(-1, 1)
            model_inputs_for_predict = model_inputs[:-1]
        
        pred_pos, pred_neg = self.predict(model_inputs_for_predict, neg_samples, node_feats, edge_types)

        # 损失计算逻辑与原来相同
        all_pred = torch.cat((pred_pos, pred_neg), dim=0)
        all_edge_label = torch.squeeze(
            torch.cat((pos_edge_label, torch.zeros_like(pos_edge_label)), dim=0)
        )
        loss = self.criterion(all_pred, all_edge_label).mean()

        return loss, all_pred, all_edge_label

    def predict(self, model_inputs, neg_samples, node_feats, edge_types=None):  # 🆕 NEW: 新增edge_types参数
        """
        预测方法 - 与HeteroSTHN_Interface的predict方法相同
        """
        # 处理model_inputs中的边类型信息
        if len(model_inputs) == 5:
            edge_feats, edge_ts, batch_size, inds, input_edge_types = model_inputs
            edge_types = input_edge_types if edge_types is None else edge_types
            model_inputs_for_base = [edge_feats, edge_ts, batch_size, inds]
        else:
            model_inputs_for_base = model_inputs
        
        # 特征提取
        if self.time_feats_dim > 0 and self.node_feats_dim == 0:
            x = self.base_model(*model_inputs_for_base, edge_types)
        elif self.time_feats_dim > 0 and self.node_feats_dim > 0:
            x = self.base_model(*model_inputs_for_base, edge_types)
            x = torch.cat([x, node_feats], dim=1)
        elif self.time_feats_dim == 0 and self.node_feats_dim > 0:
            x = node_feats
        else:
            logging.info("Either time_feats_dim or node_feats_dim must larger than 0!")

        pred_pos, pred_neg = self.edge_predictor(x, neg_samples=neg_samples, edge_types=edge_types)
        return pred_pos, pred_neg