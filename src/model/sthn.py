import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import numpy as np
from torch import Tensor
import logging
from torch_geometric.data import Data

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
        # x : [ batch_size, graph_size, edge_dims+time_dims]

        edge_time_feats = self.feat_encoder(edge_feats, edge_ts)


        x = torch.zeros(
            (batch_size * self.per_graph_size, edge_time_feats.size(1)),
            device=edge_feats.device,
        )

        x[inds] = x[inds] + edge_time_feats

        x = x.view(
            -1, self.per_graph_size // self.window_size, self.window_size * x.shape[-1]
        )

        x = self.pad_projector(x)

        x = self.p_enc_1d_model_sum(x)

        for i in range(self.num_layers):

            x = self.mixer_blocks[i](x)

        x = self.layernorm(x)

        x = torch.mean(x, dim=1)

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

        self.criterion = nn.BCEWithLogitsLoss(reduction="mean")
        self.reset_parameters()

    def reset_parameters(self):
        if self.time_feats_dim > 0:
            self.base_model.reset_parameters()
        self.edge_predictor.reset_parameters()

    def forward(self, model_inputs, neg_samples, node_feats):
        print("len(model_inputs): ", len(model_inputs))
        pred_pos, pred_neg = self.predict(model_inputs, neg_samples, node_feats)
        all_pred_logits = torch.cat((pred_pos, pred_neg), dim=0)
        all_edge_label = torch.cat(
            (torch.ones_like(pred_pos), torch.zeros_like(pred_neg)), dim=0
        )
        loss = self.criterion(all_pred_logits, all_edge_label).mean()


        all_pred_prob = torch.sigmoid(all_pred_logits)

        return loss, all_pred_prob, all_edge_label

    def predict(self, model_inputs, neg_samples, node_feats):
        if self.time_feats_dim > 0 and self.node_feats_dim == 0:
            x = self.base_model(*model_inputs)
        elif self.time_feats_dim > 0 and self.node_feats_dim > 0:
            print("len(model_inputs): ", len(model_inputs))
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


        all_pred_logits = torch.cat((pred_pos, pred_neg), dim=0)
        all_edge_label = torch.cat(
            (torch.ones_like(pred_pos), torch.zeros_like(pred_neg)), dim=0
        )
        loss = self.criterion(all_pred_logits, all_edge_label).mean()


        all_pred_prob = torch.sigmoid(all_pred_logits)

        return loss, all_pred_prob, all_edge_label

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

    def __init__(self, edge_types: list, time_dim: int = 100):
        super(HeteroTimeEncode, self).__init__()
        self.edge_types = edge_types
        self.time_dim = time_dim
        self.num_edge_types = len(edge_types)


        self.type_encoders = nn.ModuleDict()
        for i, edge_type in enumerate(edge_types):
            encoder = TimeEncode(time_dim)


            freq_multiplier = 1.0 + i * 0.1
            encoder.w.weight.data *= freq_multiplier
            self.type_encoders[str(edge_type)] = encoder


        self.default_encoder = TimeEncode(time_dim)

    def forward(self, edge_ts: torch.Tensor, edge_types: torch.Tensor = None):
        if edge_types is None:

            return self.default_encoder(edge_ts)


        batch_size = edge_ts.shape[0]
        time_embeddings = torch.zeros(
            batch_size, self.time_dim, device=edge_ts.device, dtype=edge_ts.dtype
        )


        for i, edge_type in enumerate(self.edge_types):

            type_mask = edge_types == i
            if type_mask.any():

                type_times = edge_ts[type_mask]

                type_encoder = self.type_encoders[str(edge_type)]
                type_embeddings = type_encoder(type_times)

                time_embeddings[type_mask] = type_embeddings

        return time_embeddings

    def reset_parameters(self):
        for encoder in self.type_encoders.values():
            encoder.reset_parameters()
        self.default_encoder.reset_parameters()


class HeteroFeatEncode(nn.Module):

    def __init__(self, edge_types: list, time_dims: int, feat_dims: int, out_dims: int):
        super(HeteroFeatEncode, self).__init__()
        self.edge_types = edge_types
        self.time_dims = time_dims
        self.feat_dims = feat_dims
        self.out_dims = out_dims


        self.time_encoder = HeteroTimeEncode(edge_types, time_dims)


        self.feat_encoders = nn.ModuleDict()
        for edge_type in edge_types:
            self.feat_encoders[str(edge_type)] = nn.Linear(
                time_dims + feat_dims, out_dims
            )


        self.edge_type_embedding = nn.Embedding(len(edge_types), out_dims)


        self.default_feat_encoder = nn.Linear(time_dims + feat_dims, out_dims)

        self.reset_parameters()

    def reset_parameters(self):
        self.time_encoder.reset_parameters()

        for encoder in self.feat_encoders.values():
            encoder.reset_parameters()

        self.edge_type_embedding.reset_parameters()

        self.default_feat_encoder.reset_parameters()

    def forward(
        self,
        edge_feats: torch.Tensor,
        edge_ts: torch.Tensor,
        edge_types: torch.Tensor = None,
    ):

        edge_time_feats = self.time_encoder(edge_ts, edge_types)


        combined_feats = torch.cat([edge_feats, edge_time_feats], dim=1)

        if edge_types is None:

            return self.default_feat_encoder(combined_feats)


        output_feats = torch.zeros(
            len(edge_feats), self.out_dims, device=edge_feats.device
        )

        for i, edge_type in enumerate(self.edge_types):
            type_mask = edge_types == i
            if type_mask.any():

                type_feats = combined_feats[type_mask]
                type_output = self.feat_encoders[str(edge_type)](type_feats)


                type_emb = self.edge_type_embedding(
                    torch.tensor(i, device=edge_feats.device)
                )
                type_output = type_output + type_emb.unsqueeze(0).expand(
                    type_output.size(0), -1
                )

                output_feats[type_mask] = type_output

        return output_feats


class HeteroPatch_Encoding(nn.Module):

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
        edge_types: list = None,
        module_spec=None,
        use_single_layer=False,
    ):
        super().__init__()
        self.per_graph_size = per_graph_size
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.edge_types = edge_types or ["0"]


        self.feat_encoder = HeteroFeatEncode(
            self.edge_types, time_channels, input_channels, hidden_channels
        )


        self.layernorm = nn.LayerNorm(hidden_channels)
        self.mlp_head = nn.Linear(hidden_channels, out_channels)


        self.mixer_blocks = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.mixer_blocks.append(
                TransformerBlock(
                    hidden_channels,
                    channel_expansion_factor,
                    dropout,
                    module_spec=module_spec,
                    use_single_layer=use_single_layer,
                )
            )


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
        self.pad_projector.reset_parameters()

    def forward(
        self, edge_feats, edge_ts, batch_size, inds, edge_types=None
    ):

        edge_time_feats = self.feat_encoder(edge_feats, edge_ts, edge_types)




        x = torch.zeros(
            (batch_size * self.per_graph_size, edge_time_feats.size(1)),
            device=edge_feats.device,
        )

        x[inds] = x[inds] + edge_time_feats


        x = x.view(
            -1, self.per_graph_size // self.window_size, self.window_size * x.shape[-1]
        )

        x = self.pad_projector(x)

        x = self.p_enc_1d_model_sum(x)


        for i in range(self.num_layers):

            x = self.mixer_blocks[i](x)


        x = self.layernorm(x)

        x = torch.mean(x, dim=1)

        x = self.mlp_head(x)
        return x


class HeteroEdgePredictor_per_node(torch.nn.Module):

    def __init__(
        self, dim_in_time, dim_in_node, predict_class, edge_types: list = None
    ):
        super().__init__()

        self.dim_in_time = dim_in_time
        self.dim_in_node = dim_in_node
        self.predict_class = predict_class
        self.edge_types = edge_types or ["0"]


        self.predictors = nn.ModuleDict()
        for edge_type in self.edge_types:
            self.predictors[str(edge_type)] = nn.ModuleDict(
                {
                    "src_fc": torch.nn.Linear(dim_in_time + dim_in_node, 100),
                    "dst_fc": torch.nn.Linear(dim_in_time + dim_in_node, 100),
                    "out_fc": torch.nn.Linear(100, predict_class),
                }
            )


        self.default_src_fc = torch.nn.Linear(dim_in_time + dim_in_node, 100)
        self.default_dst_fc = torch.nn.Linear(dim_in_time + dim_in_node, 100)
        self.default_out_fc = torch.nn.Linear(100, predict_class)

        self.reset_parameters()

    def reset_parameters(self):

        for predictor_dict in self.predictors.values():
            for layer in predictor_dict.values():
                layer.reset_parameters()


        self.default_src_fc.reset_parameters()
        self.default_dst_fc.reset_parameters()
        self.default_out_fc.reset_parameters()

    def forward(self, h, neg_samples=1, edge_types=None):
        num_edge = h.shape[0] // (neg_samples + 2)
        h_src = h[:num_edge]
        h_pos_dst = h[num_edge : 2 * num_edge]
        h_neg_dst = h[2 * num_edge :]
        h_save = h[: 2 * num_edge]
        if edge_types is None or len(self.edge_types) == 1:

            if len(self.edge_types) == 1:

                predictor = self.predictors[self.edge_types[0]]
                h_src_enc = predictor["src_fc"](h_src)
                h_pos_dst_enc = predictor["dst_fc"](h_pos_dst)
                h_neg_dst_enc = predictor["dst_fc"](h_neg_dst)

                h_pos_edge = torch.nn.functional.relu(h_src_enc + h_pos_dst_enc)
                h_neg_edge = torch.nn.functional.relu(
                    h_src_enc.tile(neg_samples, 1) + h_neg_dst_enc
                )

                return (
                    predictor["out_fc"](h_pos_edge),
                    predictor["out_fc"](h_neg_edge),
                    h_save,
                )
            else:

                h_src_enc = self.default_src_fc(h_src)
                h_pos_dst_enc = self.default_dst_fc(h_pos_dst)
                h_neg_dst_enc = self.default_dst_fc(h_neg_dst)

                h_pos_edge = torch.nn.functional.relu(h_src_enc + h_pos_dst_enc)
                h_neg_edge = torch.nn.functional.relu(
                    h_src_enc.tile(neg_samples, 1) + h_neg_dst_enc
                )

                return (
                    self.default_out_fc(h_pos_edge),
                    self.default_out_fc(h_neg_edge),
                    h_save,
                )

        else:

            pred_pos, pred_neg = self._hetero_forward(
                h_src, h_pos_dst, h_neg_dst, edge_types, neg_samples
            )
            return pred_pos, pred_neg, h_save

    def _hetero_forward(self, h_src, h_pos_dst, h_neg_dst, edge_types, neg_samples):
        num_edge = h_src.shape[0]

        if len(edge_types) < num_edge:

            rand_fill = np.random.randint(
                0, len(self.edge_types), num_edge - len(edge_types)
            )
            edge_types = torch.cat(
                [edge_types, torch.tensor(rand_fill, device=edge_types.device)], dim=0
            )


        pos_preds = []
        neg_preds = []


        for i, edge_type in enumerate(self.edge_types):
            type_mask = edge_types[:num_edge] == i

            if type_mask.any():
                predictor = self.predictors[str(edge_type)]


                type_h_src = h_src[type_mask]
                type_h_pos_dst = h_pos_dst[type_mask]


                type_h_src_enc = predictor["src_fc"](type_h_src)
                type_h_pos_dst_enc = predictor["dst_fc"](type_h_pos_dst)


                type_neg_indices = []
                for pos_idx in torch.where(type_mask)[0]:
                    neg_start = pos_idx * neg_samples
                    neg_end = (pos_idx + 1) * neg_samples
                    type_neg_indices.extend(range(neg_start, neg_end))

                if type_neg_indices:
                    type_h_neg_dst = h_neg_dst[type_neg_indices]
                    type_h_neg_dst_enc = predictor["dst_fc"](type_h_neg_dst)


                    type_h_pos_edge = torch.nn.functional.relu(
                        type_h_src_enc + type_h_pos_dst_enc
                    )
                    type_h_neg_edge = torch.nn.functional.relu(
                        type_h_src_enc.repeat_interleave(neg_samples, dim=0)
                        + type_h_neg_dst_enc
                    )


                    type_pos_pred = predictor["out_fc"](type_h_pos_edge)
                    type_neg_pred = predictor["out_fc"](type_h_neg_edge)

                    pos_preds.append(type_pos_pred)
                    neg_preds.append(type_neg_pred)


        if pos_preds:
            return torch.cat(pos_preds, dim=0), torch.cat(neg_preds, dim=0)
        else:

            device = h_src.device
            return torch.empty(0, self.predict_class, device=device), torch.empty(
                0, self.predict_class, device=device
            )


class HeteroSTHN_Interface(nn.Module):

    def __init__(
        self, mlp_mixer_configs, edge_predictor_configs, edge_types: list = None
    ):
        super(HeteroSTHN_Interface, self).__init__()

        self.time_feats_dim = edge_predictor_configs["dim_in_time"]
        self.node_feats_dim = edge_predictor_configs["dim_in_node"]
        self.edge_types = edge_types or ["0"]


        if self.time_feats_dim > 0:

            mlp_mixer_configs["edge_types"] = self.edge_types
            self.base_model = HeteroPatch_Encoding(
                **mlp_mixer_configs
            )


        edge_predictor_configs["edge_types"] = self.edge_types
        self.edge_predictor = HeteroEdgePredictor_per_node(
            **edge_predictor_configs
        )


        self.criterion = nn.BCEWithLogitsLoss(reduction="mean")
        self.reset_parameters()

    def reset_parameters(self):
        if self.time_feats_dim > 0:
            self.base_model.reset_parameters()
        self.edge_predictor.reset_parameters()

    def forward(
        self, model_inputs, neg_samples, node_feats
    ):
        edge_feats = model_inputs[0]

        edge_types = torch.argmax(edge_feats, dim=1)
        pred_pos, pred_neg = self.predict(
            model_inputs, neg_samples, node_feats, edge_types
        )


        all_pred_logits = torch.cat((pred_pos, pred_neg), dim=0)
        all_edge_label = torch.cat(
            (torch.ones_like(pred_pos), torch.zeros_like(pred_neg)), dim=0
        )
        loss = self.criterion(all_pred_logits, all_edge_label).mean()


        all_pred_prob = torch.sigmoid(all_pred_logits)

        return loss, all_pred_prob, all_edge_label

    def predict(
        self, model_inputs, neg_samples, node_feats, edge_types=None
    ):

        if len(model_inputs) == 5:

            edge_feats, edge_ts, batch_size, inds, input_edge_types = model_inputs

            edge_types = input_edge_types if edge_types is None else edge_types

            model_inputs_for_base = [edge_feats, edge_ts, batch_size, inds]
        else:

            model_inputs_for_base = model_inputs


        if self.time_feats_dim > 0 and self.node_feats_dim == 0:

            x = self.base_model(*model_inputs_for_base, edge_types)
        elif self.time_feats_dim > 0 and self.node_feats_dim > 0:

            x = self.base_model(*model_inputs_for_base, edge_types)
            x = torch.cat([x, node_feats], dim=1)
        elif self.time_feats_dim == 0 and self.node_feats_dim > 0:
            x = node_feats
        else:
            logging.info("Either time_feats_dim or node_feats_dim must larger than 0!")


        pred_pos, pred_neg = self.edge_predictor(
            x, neg_samples=neg_samples, edge_types=edge_types
        )
        return pred_pos, pred_neg


class HeteroMulticlass_Interface(nn.Module):

    def __init__(
        self, mlp_mixer_configs, edge_predictor_configs, edge_types: list = None
    ):
        super(HeteroMulticlass_Interface, self).__init__()

        self.time_feats_dim = edge_predictor_configs["dim_in_time"]
        self.node_feats_dim = edge_predictor_configs["dim_in_node"]
        self.edge_types = edge_types or ["0"]


        if self.time_feats_dim > 0:
            mlp_mixer_configs["edge_types"] = self.edge_types
            self.base_model = HeteroPatch_Encoding(**mlp_mixer_configs)

        edge_predictor_configs["edge_types"] = self.edge_types
        self.edge_predictor = HeteroEdgePredictor_per_node(**edge_predictor_configs)


        self.criterion = nn.CrossEntropyLoss(reduction="mean")
        self.reset_parameters()

    def reset_parameters(self):
        if self.time_feats_dim > 0:
            self.base_model.reset_parameters()
        self.edge_predictor.reset_parameters()

    def forward(
        self, model_inputs, neg_samples, node_feats, edge_types=None
    ):

        if (
            len(model_inputs) == 6
        ):  # [edge_feats, edge_ts, batch_size, inds, pos_edge_label, edge_types]
            pos_edge_label = model_inputs[-2].view(-1, 1)
            edge_types = model_inputs[-1] if edge_types is None else edge_types
            model_inputs_for_predict = model_inputs[:-2]
        else:
            pos_edge_label = model_inputs[-1].view(-1, 1)
            model_inputs_for_predict = model_inputs[:-1]

        pred_pos, pred_neg = self.predict(
            model_inputs_for_predict, neg_samples, node_feats, edge_types
        )


        all_pred_logits = torch.cat((pred_pos, pred_neg), dim=0)
        all_edge_label = torch.cat(
            (torch.ones_like(pred_pos), torch.zeros_like(pred_neg)), dim=0
        )
        loss = self.criterion(all_pred_logits, all_edge_label).mean()


        all_pred_prob = torch.sigmoid(all_pred_logits)

        return loss, all_pred_prob, all_edge_label

    def predict(
        self, model_inputs, neg_samples, node_feats, edge_types=None
    ):

        if len(model_inputs) == 6:
            edge_feats, edge_ts, batch_size, inds, input_edge_types = model_inputs
            edge_types = input_edge_types if edge_types is None else edge_types
            model_inputs_for_base = [edge_feats, edge_ts, batch_size, inds]
        else:
            model_inputs_for_base = model_inputs


        if self.time_feats_dim > 0 and self.node_feats_dim == 0:
            x = self.base_model(*model_inputs_for_base, edge_types)
        elif self.time_feats_dim > 0 and self.node_feats_dim > 0:
            x = self.base_model(*model_inputs_for_base, edge_types)
            x = torch.cat([x, node_feats], dim=1)
        elif self.time_feats_dim == 0 and self.node_feats_dim > 0:
            x = node_feats
        else:
            logging.info("Either time_feats_dim or node_feats_dim must larger than 0!")

        pred_pos, pred_neg = self.edge_predictor(
            x, neg_samples=neg_samples, edge_types=edge_types
        )
        return pred_pos, pred_neg


from riemanngfm.modules.model import GeoGFM


class RiemannianStructuralEncoder(nn.Module):
    def __init__(
        self, n_layers, in_dim, hidden_dim, embed_dim, bias, activation, dropout
    ):
        super().__init__()

        self.gfgm_model = GeoGFM(
            n_layers, in_dim, hidden_dim, embed_dim, bias, activation, dropout
        )

    def forward(self, structural_data):


        x_E, x_H, x_S = self.gfgm_model(structural_data)


        manifold_H = self.gfgm_model.manifold_H
        manifold_S = self.gfgm_model.manifold_S
        x_h_tangent = manifold_H.logmap0(x_H)
        x_s_tangent = manifold_S.logmap0(x_S)



        structural_embedding = torch.cat([x_E, x_h_tangent, x_s_tangent], dim=-1)

        return structural_embedding

    def reset_parameters(self):

        for module in self.gfgm_model.modules():

            if hasattr(module, "reset_parameters"):

                module.reset_parameters()


class HeteroSTHN_Interface_rgfm(nn.Module):

    def __init__(
        self,
        mlp_mixer_configs: dict,
        edge_predictor_configs: dict,
        edge_types: list = None,
        riemannian_configs: dict = None,
    ):
        super(HeteroSTHN_Interface_rgfm, self).__init__()

        self.time_feats_dim = edge_predictor_configs["dim_in_time"]
        self.node_feats_dim = edge_predictor_configs["dim_in_node"]
        self.edge_types = edge_types or ["0"]


        if self.time_feats_dim > 0:
            mlp_mixer_configs["edge_types"] = self.edge_types
            self.base_model = HeteroPatch_Encoding(**mlp_mixer_configs)


        edge_predictor_configs["edge_types"] = self.edge_types
        self.edge_predictor = HeteroEdgePredictor_per_node(**edge_predictor_configs)


        self.criterion = nn.BCEWithLogitsLoss(reduction="mean")


        self.use_riemannian = riemannian_configs is not None
        if self.use_riemannian:
            self.riemannian_encoder = RiemannianStructuralEncoder(**riemannian_configs)


            temporal_dim = mlp_mixer_configs.get("out_channels", 0)
            if self.node_feats_dim > 0:
                temporal_dim += self.node_feats_dim

            structural_dim = 3 * riemannian_configs.get("embed_dim", 0)

            self.dynamic_alignment = DynamicAlignmentLayer(structural_dim)

            predictor_input_dim = (
                edge_predictor_configs["dim_in_time"]
                + edge_predictor_configs["dim_in_node"]
            )

            self.fusion_layer = nn.Sequential(
                nn.Linear(temporal_dim + structural_dim, predictor_input_dim * 2),
                nn.ReLU(),
                nn.Linear(predictor_input_dim * 2, predictor_input_dim),
            )

        self.reset_parameters()

    def reset_parameters(self):
        if self.time_feats_dim > 0:
            self.base_model.reset_parameters()
        self.edge_predictor.reset_parameters()
        if self.use_riemannian:
            self.riemannian_encoder.reset_parameters()
            for layer in self.fusion_layer:
                if isinstance(layer, nn.Linear):
                    layer.reset_parameters()

    def forward(
        self,
        model_inputs,
        neg_samples,
        node_feats,

        structural_data: Data = None,
    ):

        edge_feats = model_inputs[0]
        edge_types = (
            torch.argmax(edge_feats, dim=1)
            if edge_feats.ndim == 2 and edge_feats.shape[1] > 1
            else None
        )

        pred_pos, pred_neg, h_save = self.predict(
            model_inputs, neg_samples, node_feats, edge_types, structural_data
        )


        all_pred_logits = torch.cat((pred_pos, pred_neg), dim=0)
        all_edge_label = torch.cat(
            (torch.ones_like(pred_pos), torch.zeros_like(pred_neg)), dim=0
        )
        loss = self.criterion(all_pred_logits, all_edge_label).mean()


        all_pred_prob = torch.sigmoid(all_pred_logits)

        return loss, all_pred_prob, all_edge_label, h_save

    def predict(
        self,
        model_inputs,
        neg_samples,
        node_feats,
        edge_types=None,

        structural_data: Data = None,
    ):

        model_inputs_for_base = model_inputs[:4]


        x_temporal = None
        if self.time_feats_dim > 0:

            x_temporal = self.base_model(*model_inputs_for_base, edge_types)

        if node_feats is not None and self.node_feats_dim > 0:
            x_temporal = (
                torch.cat([x_temporal, node_feats], dim=1)
                if x_temporal is not None
                else node_feats
            )


        if self.use_riemannian and structural_data is not None:
            if x_temporal is None:
                raise ValueError(
                    "Temporal features must be computed to be fused with structural features."
                )

            z_struct = self.riemannian_encoder(structural_data)
            # aligned_z_struct = z_struct[structural_data.root_nodes_mask]

            target_batch_size = x_temporal.shape[0]
            aligned_z_struct = self.dynamic_alignment(z_struct, target_batch_size)
            final_x = torch.cat([x_temporal, aligned_z_struct], dim=1)
            final_x = self.fusion_layer(final_x)
        else:
            final_x = x_temporal

        if final_x is None:
            raise ValueError(
                "No features were generated. Check your model's feature dimension settings."
            )


        pred_pos, pred_neg, _ = self.edge_predictor(
            final_x, neg_samples=neg_samples, edge_types=edge_types
        )
        num_edge = aligned_z_struct.shape[0] // (neg_samples + 2)
        # h_save = aligned_z_struct[: 2 * num_edge]
        h_save = final_x[: 2 * num_edge]
        return pred_pos, pred_neg, h_save


class DynamicAlignmentLayer(nn.Module):

    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim

        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim), nn.Tanh(), nn.Linear(feature_dim, 1)
        )

    def forward(self, z_struct, target_length):
        source_length = z_struct.shape[0]

        if source_length == target_length:
            return z_struct
        elif source_length > target_length:

            attention_weights = self.attention(z_struct)  # [source_length, 1]
            attention_weights = torch.softmax(
                attention_weights.squeeze(-1), dim=0
            )  # [source_length]


            _, top_indices = torch.topk(attention_weights, target_length)
            top_indices = torch.sort(top_indices)[0]
            return z_struct[top_indices]
        else:


            repeat_times = (target_length + source_length - 1) // source_length
            z_repeated = z_struct.repeat(repeat_times, 1)[:target_length]


            position_adjust = torch.arange(
                target_length, device=z_struct.device, dtype=torch.float
            )
            position_adjust = position_adjust / target_length
            position_weight = torch.sigmoid(position_adjust).unsqueeze(
                1
            )  # [target_length, 1]

            return z_repeated * position_weight


class STHN_Interface_rgfm(STHN_Interface):
    def __init__(
        self, mlp_mixer_configs, edge_predictor_configs, riemannian_configs=None
    ):
        self.use_riemannian = riemannian_configs is not None
        self.structural_dim = 0
        self.riemannian_encoder = None
        self.dynamic_alignment = None
        self.fusion_layer = None
        super().__init__(mlp_mixer_configs, edge_predictor_configs)
        if self.use_riemannian:
            embed_dim = riemannian_configs.get("embed_dim", 0)
            self.structural_dim = 3 * embed_dim
            self.riemannian_encoder = RiemannianStructuralEncoder(**riemannian_configs)
            self.dynamic_alignment = DynamicAlignmentLayer(self.structural_dim)
            temporal_dim = mlp_mixer_configs.get("out_channels", 0)
            if self.node_feats_dim > 0:
                temporal_dim += self.node_feats_dim
            predictor_dim = (
                edge_predictor_configs["dim_in_time"]
                + edge_predictor_configs["dim_in_node"]
            )
            self.fusion_layer = nn.Sequential(
                nn.Linear(temporal_dim + self.structural_dim, predictor_dim * 2),
                nn.ReLU(),
                nn.Linear(predictor_dim * 2, predictor_dim),
            )

    def reset_parameters(self):
        super().reset_parameters()
        if self.use_riemannian and self.riemannian_encoder is not None:
            self.riemannian_encoder.reset_parameters()
        if self.fusion_layer is not None:
            for layer in self.fusion_layer:
                if isinstance(layer, nn.Linear):
                    layer.reset_parameters()

    def forward(
        self, model_inputs, neg_samples, node_feats, structural_data: Data = None
    ):
        pred_pos, pred_neg = self.predict(
            model_inputs, neg_samples, node_feats, structural_data=structural_data
        )
        all_pred_logits = torch.cat((pred_pos, pred_neg), dim=0)
        all_edge_label = torch.cat(
            (torch.ones_like(pred_pos), torch.zeros_like(pred_neg)), dim=0
        )
        loss = self.criterion(all_pred_logits, all_edge_label).mean()
        all_pred_prob = torch.sigmoid(all_pred_logits)
        return loss, all_pred_prob, all_edge_label

    def predict(
        self, model_inputs, neg_samples, node_feats, structural_data: Data = None
    ):
        if self.time_feats_dim > 0 and self.node_feats_dim == 0:
            x = self.base_model(*model_inputs)
        elif self.time_feats_dim > 0 and self.node_feats_dim > 0:
            x = self.base_model(*model_inputs)
            x = torch.cat([x, node_feats], dim=1)
        elif self.time_feats_dim == 0 and self.node_feats_dim > 0:
            x = node_feats
        else:
            raise ValueError(
                "Either time_feats_dim or node_feats_dim must be larger than 0."
            )

        if self.use_riemannian and structural_data is not None:
            z_struct = self.riemannian_encoder(structural_data)
            aligned_struct = self.dynamic_alignment(z_struct, x.size(0))
            x = torch.cat([x, aligned_struct], dim=1)
            x = self.fusion_layer(x)

        return self.edge_predictor(x, neg_samples=neg_samples)


class HeteroSTHN_Interface_rgfm_loss(nn.Module):

    def __init__(
        self,
        mlp_mixer_configs: dict,
        edge_predictor_configs: dict,
        edge_types: list = None,
        riemannian_configs: dict = None,
        alpha: float = 0.2,
    ):
        super(HeteroSTHN_Interface_rgfm_loss, self).__init__()

        self.time_feats_dim = edge_predictor_configs["dim_in_time"]
        self.node_feats_dim = edge_predictor_configs[
            "dim_in_node"
        ]
        self.edge_types = edge_types or ["0"]
        self.alpha = alpha


        if self.time_feats_dim > 0:
            mlp_mixer_configs["edge_types"] = self.edge_types
            self.base_model = HeteroPatch_Encoding(**mlp_mixer_configs)


        edge_predictor_configs["edge_types"] = self.edge_types
        self.edge_predictor = HeteroEdgePredictor_per_node(**edge_predictor_configs)


        self.criterion = nn.BCEWithLogitsLoss(reduction="mean")


        self.align_loss_fn = SymmetricAlignmentLoss()


        align_dim = mlp_mixer_configs.get("out_channels", 128)


        t_dim = mlp_mixer_configs.get("out_channels", 0)
        self.proj_t = (
            nn.Linear(t_dim, align_dim) if t_dim != align_dim else nn.Identity()
        )


        self.use_riemannian = riemannian_configs is not None
        if self.use_riemannian:
            self.riemannian_encoder = RiemannianStructuralEncoder(**riemannian_configs)


            structural_dim = 3 * riemannian_configs.get("embed_dim", 0)
            self.dynamic_alignment = DynamicAlignmentLayer(structural_dim)


            self.proj_s = nn.Linear(structural_dim, align_dim)


            fusion_input_dim = t_dim + structural_dim


            predictor_input_dim = self.time_feats_dim + self.node_feats_dim


            self.fusion_layer = nn.Sequential(
                nn.Linear(fusion_input_dim, predictor_input_dim * 2),
                nn.ReLU(),
                nn.Linear(predictor_input_dim * 2, predictor_input_dim),
            )

        self.reset_parameters()

    def reset_parameters(self):
        if self.time_feats_dim > 0:
            self.base_model.reset_parameters()
        self.edge_predictor.reset_parameters()

        if hasattr(self, "proj_t") and isinstance(self.proj_t, nn.Linear):
            self.proj_t.reset_parameters()

        if hasattr(self, "proj_s") and isinstance(self.proj_s, nn.Linear):
            self.proj_s.reset_parameters()

        if self.use_riemannian:
            self.riemannian_encoder.reset_parameters()
            for layer in self.fusion_layer:
                if isinstance(layer, nn.Linear):
                    layer.reset_parameters()

    def forward(
        self, model_inputs, neg_samples, node_feats, structural_data: Data = None
    ):

        edge_feats = model_inputs[0]
        edge_types = (
            torch.argmax(edge_feats, dim=1)
            if edge_feats.ndim == 2 and edge_feats.shape[1] > 1
            else None
        )



        pred_pos, pred_neg, z_t, z_s = self.predict(
            model_inputs, neg_samples, node_feats, edge_types, structural_data
        )


        all_pred_logits = torch.cat((pred_pos, pred_neg), dim=0)
        all_edge_label = torch.cat(
            (torch.ones_like(pred_pos), torch.zeros_like(pred_neg)), dim=0
        )
        loss_main = self.criterion(all_pred_logits, all_edge_label).mean()


        loss_align = torch.tensor(0.0, device=loss_main.device)
        if z_t is not None and z_s is not None:

            z_t_proj = self.proj_t(z_t)
            z_s_proj = self.proj_s(z_s)

            loss_align = self.align_loss_fn(z_t_proj, z_s_proj)


        total_loss = loss_main + self.alpha * loss_align

        all_pred_prob = torch.sigmoid(all_pred_logits)

        num_edge = z_s.shape[0] // (neg_samples + 2)
        z_s = z_s[: 2 * num_edge]
        z_t = z_t[: 2 * num_edge]

        return total_loss, all_pred_prob, all_edge_label, torch.cat([z_s, z_t], dim=1)

    def predict(
        self,
        model_inputs,
        neg_samples,
        node_feats,
        edge_types=None,
        structural_data: Data = None,
    ):

        model_inputs_for_base = model_inputs[:4]


        x_temporal = None
        if self.time_feats_dim > 0:
            x_temporal = self.base_model(*model_inputs_for_base, edge_types)


        z_t = x_temporal

        z_s = None
        final_x = None


        if self.use_riemannian and structural_data is not None:
            if x_temporal is None:
                raise ValueError(
                    "Temporal features must be computed to be fused with structural features."
                )


            raw_z_struct = self.riemannian_encoder(structural_data)


            target_batch_size = x_temporal.shape[0]
            aligned_z_struct = self.dynamic_alignment(raw_z_struct, target_batch_size)


            z_s = aligned_z_struct



            fusion_input = torch.cat([x_temporal, aligned_z_struct], dim=1)
            final_x = self.fusion_layer(fusion_input)
        else:

            final_x = x_temporal

        if final_x is None:
            raise ValueError("No features were generated.")


        pred_pos, pred_neg, _ = self.edge_predictor(
            final_x, neg_samples=neg_samples, edge_types=edge_types
        )

        return pred_pos, pred_neg, z_t, z_s


class SymmetricAlignmentLoss(nn.Module):

    def __init__(self):
        super(SymmetricAlignmentLoss, self).__init__()

    def forward(self, z_t, z_x):


        p_t = F.softmax(z_t, dim=-1)
        p_x = F.softmax(z_x, dim=-1)

        log_p_t = F.log_softmax(z_t, dim=-1)
        log_p_x = F.log_softmax(z_x, dim=-1)







        # JSD term 1: Sum( z_t * log(z_t / z_x) ) = KL(z_t || z_x)
        kl_t_x = F.kl_div(log_p_x, p_t, reduction="none").sum(dim=-1)

        # JSD term 2: Sum( z_x * log(z_x / z_t) ) = KL(z_x || z_t)
        kl_x_t = F.kl_div(log_p_t, p_x, reduction="none").sum(dim=-1)


        # jsd_val: [batch_size]
        jsd_val = kl_t_x + kl_x_t


        total_jsd = jsd_val.sum()


        loss_align = torch.sigmoid(total_jsd)

        return loss_align
