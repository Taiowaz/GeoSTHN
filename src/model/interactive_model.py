import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.sthn import STHN_Interface


class AttentionFusionLayer(nn.Module):
    def __init__(self, query_dim, text_embedding_dim):
        """
        query_dim: h_graph (来自Patch_Encoding)的维度
        text_embedding_dim: LLM文本嵌入的维度
        """
        super().__init__()
        # 确保Query和Key的维度一致，如果不一致，需要一个线性层来投影
        # 这里我们假设 text_embedding_dim 和 query_dim 经过处理后是一致的
        # 如果不一致，可以像下面这样加一个投影层
        # self.query_projection = nn.Linear(query_dim, text_embedding_dim)
        self.scale = text_embedding_dim**-0.5

    def forward(self, h_graph_query, text_embeddings):
        """
        h_graph_query: GNN输出的h_graph, 形状 [batch_size, query_dim]
        text_embeddings: LLM文本嵌入, 形状 [batch_size, num_texts, text_embedding_dim]
                         注意：每个图样本都应该有自己的一组文本嵌入
        """
        # query = self.query_projection(h_graph_query) # 如果需要投影
        query = h_graph_query  # 假设维度已经对齐
        keys = text_embeddings
        values = text_embeddings

        # 为了进行批量矩阵乘法，调整query的维度
        query = query.unsqueeze(1)  # [batch_size, 1, query_dim]

        # 1. 计算注意力分数
        scores = (
            torch.bmm(query, keys.transpose(1, 2)) * self.scale
        )  # [batch_size, 1, num_texts]

        # 2. 计算注意力权重
        attn_weights = F.softmax(scores, dim=-1)  # [batch_size, 1, num_texts]

        # 3. 计算上下文向量
        c_text = torch.bmm(attn_weights, values).squeeze(
            1
        )  # [batch_size, text_embedding_dim]

        # 4. 融合
        h_fused = torch.cat((h_graph_query, c_text), dim=-1)

        return h_fused, attn_weights


class FusedEdgePredictor(nn.Module):
    def __init__(self, fused_dim, predict_class):
        """
        fused_dim: 拼接了文本向量后的维度 (query_dim + text_embedding_dim)
        predict_class: 预测的类别数
        """
        super().__init__()
        # 我们简化一下，直接对融合后的src和dst表示进行处理
        hidden_dim = 128  # 可以自定义
        self.src_fc = nn.Linear(fused_dim, hidden_dim)
        self.dst_fc = nn.Linear(fused_dim, hidden_dim)
        self.out_fc = nn.Linear(hidden_dim, predict_class)

    def forward(self, h_src_fused, h_pos_dst_fused, h_neg_dst_fused):
        # h_src_fused, h_pos_dst_fused 的 batch_size 应该是一样的
        # h_neg_dst_fused 的 batch_size 是前者的 neg_samples 倍

        # 正样本边
        h_src = self.src_fc(h_src_fused)
        h_pos_dst = self.dst_fc(h_pos_dst_fused)
        h_pos_edge = F.relu(h_src + h_pos_dst)
        pred_pos = self.out_fc(h_pos_edge)

        # 负样本边
        # 需要将 h_src 复制 neg_samples 次以匹配 h_neg_dst_fused
        num_neg_samples = h_neg_dst_fused.size(0) // h_src_fused.size(0)
        h_src_tiled = h_src.repeat_interleave(num_neg_samples, dim=0)
        h_neg_dst = self.dst_fc(h_neg_dst_fused)
        h_neg_edge = F.relu(h_src_tiled + h_neg_dst)
        pred_neg = self.out_fc(h_neg_edge)

        return pred_pos, pred_neg


class InteractiveSTHN(nn.Module):
    def __init__(self, sthn_model: STHN_Interface, text_embedding_dim: int):
        super().__init__()
        # 保留你原来的STHN模型，作为基础特征提取器
        self.sthn_model = sthn_model

        # 获取GNN输出的维度
        # 这需要从你的配置中得知，或者直接查看 sthn_model 的输入维度
        # 假设 sthn_model.time_feats_dim + sthn_model.node_feats_dim 是最终进入predictor的维度
        # 我们需要看 Patch_Encoding 的输出维度
        # 从代码看，Patch_Encoding 的输出是 mlp_mixer_configs['out_channels']
        query_dim = sthn_model.base_model.mlp_head.in_features  # 这是更准确的获取方式

        # 初始化我们的新模块
        self.attention_layer = AttentionFusionLayer(query_dim, text_embedding_dim)

        fused_dim = query_dim + text_embedding_dim
        predict_class = sthn_model.edge_predictor.out_fc.out_features
        self.fused_predictor = FusedEdgePredictor(fused_dim, predict_class)

        self.creterion = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, model_inputs, neg_samples, node_feats, text_embeddings):
        """
        text_embeddings: 形状为 [batch_size, num_texts, text_embedding_dim]
                         注意：这里的batch_size对应的是(src, pos_dst, neg_dst)的总数
        """

        # --- 1. 从原模型获取 h_graph ---
        # 我们需要稍微修改一下原模型的逻辑，让它只返回向量x，而不是直接预测
        if self.sthn_model.time_feats_dim > 0 and self.sthn_model.node_feats_dim == 0:
            x = self.sthn_model.base_model(*model_inputs)
        elif self.sthn_model.time_feats_dim > 0 and self.sthn_model.node_feats_dim > 0:
            base_x = self.sthn_model.base_model(*model_inputs)
            x = torch.cat([base_x, node_feats], dim=1)
        elif self.sthn_model.time_feats_dim == 0 and self.sthn_model.node_feats_dim > 0:
            x = node_feats
        else:
            raise ValueError(
                "Either time_feats_dim or node_feats_dim must larger than 0!"
            )

        # --- 2. 分离 src, pos_dst, neg_dst ---
        num_edge = x.shape[0] // (neg_samples + 2)
        h_src = x[:num_edge]
        h_pos_dst = x[num_edge : 2 * num_edge]
        h_neg_dst = x[2 * num_edge :]

        # --- 3. 为每个部分应用注意力融合 ---
        # 假设每个src/dst/neg_dst都对应同一份LLM对该(src,dst)关系的分析文本
        # 因此，文本嵌入需要被正确地广播
        num_src = h_src.shape[0]
        text_emb_src = text_embeddings[:num_src]
        text_emb_pos_dst = text_embeddings[:num_src]  # 正样本目标节点也用同样的文本分析
        text_emb_neg_dst = text_embeddings[:num_src].repeat_interleave(
            neg_samples, dim=0
        )

        h_src_fused, _ = self.attention_layer(h_src, text_emb_src)
        h_pos_dst_fused, _ = self.attention_layer(h_pos_dst, text_emb_pos_dst)
        h_neg_dst_fused, _ = self.attention_layer(h_neg_dst, text_emb_neg_dst)

        # --- 4. 使用新的预测器进行预测 ---
        pred_pos, pred_neg = self.fused_predictor(
            h_src_fused, h_pos_dst_fused, h_neg_dst_fused
        )

        # --- 5. 计算损失 ---
        # 这部分与你原来的逻辑保持一致
        all_pred = torch.cat((pred_pos, pred_neg), dim=0)
        # 根据你的任务是二分类还是多分类，创建正确的label
        # 以二分类为例
        pos_labels = torch.ones(pred_pos.size(0), device=x.device, dtype=torch.long)
        neg_labels = torch.zeros(pred_neg.size(0), device=x.device, dtype=torch.long)
        all_labels = torch.cat((pos_labels, neg_labels), dim=0)

        # 如果是多分类，label的创建方式要相应调整
        # loss = self.creterion(all_pred, all_labels.squeeze(-1))
        # 如果你的predictor输出维度>1, CrossEntropyLoss不需要squeeze
        loss = self.creterion(all_pred, all_labels)

        return loss, all_pred, all_labels
