from re import sub
import torch
import numpy as np
import logging
from tqdm import tqdm
import time
import copy
import hashlib
from torch_sparse import SparseTensor
from torchmetrics.classification import MulticlassAUROC, MulticlassAveragePrecision
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from sklearn.preprocessing import MinMaxScaler
from src.utils.utils import evaluate_mrr
from tgb.linkproppred.evaluate import Evaluator
from scipy.sparse.linalg import ArpackError

from src.utils.construct_subgraph import (
    get_random_inds,
    get_all_inds,
    construct_mini_batch_giant_graph,
    get_subgraph_sampler,
    pre_compute_subgraphs,
)
from src.utils.utils import row_norm


def get_inputs_for_ind(
    subgraphs,
    mode,
    cached_neg_samples,
    neg_samples,
    node_feats,
    edge_feats,
    cur_df,
    df_all,
    cur_inds,
    ind,
    args,
):
    subgraphs, elabel = subgraphs
    scaler = MinMaxScaler()
    if args.use_cached_subgraph == False and mode == "train":
        subgraph_data_list = subgraphs.all_root_nodes[ind]
        mini_batch_inds = get_random_inds(
            len(subgraph_data_list), cached_neg_samples, neg_samples
        )
        subgraph_data_raw = subgraphs.mini_batch(ind, mini_batch_inds)
    elif mode in ["test", "tgb-val"]:
        assert cached_neg_samples == neg_samples
        subgraph_data_list = subgraphs[ind]
        mini_batch_inds = get_all_inds(len(subgraph_data_list), cached_neg_samples)
        subgraph_data_raw = [subgraph_data_list[i] for i in mini_batch_inds]
    else:  # sthn valid
        # 获取的是子图数据
        subgraph_data_list = subgraphs[ind]
        # [batch_size(node_index_src), batch_size(node_index_dst), batch_size(node_index_neg)]
        mini_batch_inds = get_random_inds(
            len(subgraph_data_list), cached_neg_samples, neg_samples
        )

        subgraph_data_raw = [subgraph_data_list[i] for i in mini_batch_inds]
    
    subgraph_data = construct_mini_batch_giant_graph(subgraph_data_raw, args.max_edges)
    if args.use_riemannian_structure:
        structural_data =  create_riemannian_data_snapshot(
            nodes=subgraph_data["nodes"],
            row=subgraph_data["row"],
            col=subgraph_data["col"],
            root_nodes=subgraph_data["root_nodes"],
            embed_dim=args.rgfm_embed_dim,
            device=args.device,
            dataset_name=args.dataset,
            )
    else:
        structural_data = None

    # raw edge feats
    subgraph_edge_feats = edge_feats[subgraph_data["eid"]]
    subgraph_edts = torch.from_numpy(subgraph_data["edts"]).float()
    if args.use_graph_structure and node_feats is not None:
        num_of_df_links = len(subgraph_data_list) // (cached_neg_samples + 2)
        # subgraph_node_feats = compute_sign_feats(node_feats, df, cur_inds, num_of_df_links, subgraph_data['root_nodes'], args)
        # Erfan: change this part to use masked version
        subgraph_node_feats = compute_sign_feats(
            node_feats,
            cur_df,
            cur_inds,
            num_of_df_links,
            subgraph_data["root_nodes"],
            args,
        )
        cur_inds += num_of_df_links
    else:
        subgraph_node_feats = None
    scaler.fit(subgraph_edts.reshape(-1, 1))
    subgraph_edts = (
        scaler.transform(subgraph_edts.reshape(-1, 1)).ravel().astype(np.float32) * 1000
    )
    subgraph_edts = torch.from_numpy(subgraph_edts)
    # if subgraph_edts.numel() > 0:
    #     scaler.fit(subgraph_edts.reshape(-1, 1))
    #     subgraph_edts = (
    #         scaler.transform(subgraph_edts.reshape(-1, 1)).ravel().astype(np.float32) * 1000
    #     )
    #     subgraph_edts = torch.from_numpy(subgraph_edts)

    # get mini-batch inds
    all_inds, has_temporal_neighbors = [], []

    # ignore an edge pair if (src_node, dst_node) does not have temporal neighbors
    all_edge_indptr = subgraph_data["all_edge_indptr"]

    for i in range(len(all_edge_indptr) - 1):
        num_edges = all_edge_indptr[i + 1] - all_edge_indptr[i]
        # 为每条边生成全局索引，格式为：子图索引 * max_edges + 边在子图内的索引
        all_inds.extend([(args.max_edges * i + j) for j in range(num_edges)])
        has_temporal_neighbors.append(num_edges > 0)

    if not args.predict_class:
        inputs = [
            subgraph_edge_feats.to(args.device),
            subgraph_edts.to(args.device),
            len(has_temporal_neighbors),
            torch.tensor(all_inds).long(),
        ]
    else:
        subgraph_edge_type = elabel[ind]
        inputs = [
            subgraph_edge_feats.to(args.device),
            subgraph_edts.to(args.device),
            len(has_temporal_neighbors),
            torch.tensor(all_inds).long(),
            torch.from_numpy(subgraph_edge_type).to(args.device),
        ]

    return inputs, subgraph_node_feats, cur_inds,structural_data


def run(
    model,
    optimizer,
    args,
    subgraphs,
    df,
    node_feats,
    edge_feats,
    MLAUROC,
    MLAUPRC,
    mode,
):
    time_epoch = 0
    ###################################################
    # setup modes
    cur_inds = 0
    if mode == "train":
        model.train()
        cur_df = df[args.train_mask]
        neg_samples = args.neg_samples
        cached_neg_samples = args.extra_neg_samples

    elif mode == "valid":
        model.eval()
        cur_df = df[args.val_mask]
        neg_samples = 1
        cached_neg_samples = 1

    elif mode == "test":
        ## Erfan: remove this part use TGB evaluation
        raise ("Use TGB evaluation")
        # model.eval()
        # cur_df = df[args.test_mask]
        # neg_samples = 1
        # cached_neg_samples = 1
        # cur_inds = args.val_edge_end

    train_loader = cur_df.groupby(cur_df.index // args.batch_size)
    pbar = tqdm(total=len(train_loader))
    pbar.set_description("%s mode with negative samples %d ..." % (mode, neg_samples))

    ###################################################
    # compute + training + fetch all scores
    loss_lst = []
    MLAUROC.reset()
    MLAUPRC.reset()

    for ind in range(len(train_loader)):
        ###################################################
        inputs, subgraph_node_feats, cur_inds,structual_data = get_inputs_for_ind(
            subgraphs,
            mode,
            cached_neg_samples,
            neg_samples,
            node_feats,
            edge_feats,
            cur_df,
            df, 
            cur_inds,
            ind,
            args,
        )

        start_time = time.time()
        # 将inputs, neg_samples, subgraph_node_feats转为张量
        if args.use_riemannian_structure:
            loss, pred, edge_label = model(inputs, neg_samples, subgraph_node_feats,structual_data)
        else:
            loss, pred, edge_label = model(inputs, neg_samples, subgraph_node_feats)
        if mode == "train" and optimizer != None:
            optimizer.zero_grad()
            if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                loss = loss.mean()
            loss.backward()
            optimizer.step()
        time_epoch += time.time() - start_time

        batch_auroc = MLAUROC.update(pred, edge_label.to(torch.int))
        batch_auprc = MLAUPRC.update(pred, edge_label.to(torch.int))
        if isinstance(loss, torch.Tensor):
            loss_lst.append(loss.mean().item())
        else:
            loss_lst.append(float(loss))
            
        del inputs, subgraph_node_feats, structual_data, loss, pred, edge_label
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        pbar.update(1)
    pbar.close()
    total_auroc = MLAUROC.compute()
    total_auprc = MLAUPRC.compute()
    return_loss = np.mean(loss_lst)
    logging.info(
        f"{mode} mode with time {time_epoch:.4f}, AUROC {total_auroc:.4f}, AUPRC {total_auprc:.4f}, loss {return_loss:.4f}"
    )
    return total_auroc, total_auprc, return_loss, time_epoch


def link_pred_train(model, args, g, df, node_feats, edge_feats):

    optimizer = torch.optim.RMSprop(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    ###################################################
    # get cached data
    if args.use_cached_subgraph:
        train_subgraphs = pre_compute_subgraphs(args, g, df, mode="train")
    else:
        train_subgraphs = get_subgraph_sampler(args, g, df, mode="train")

    valid_subgraphs = pre_compute_subgraphs(args, g, df, mode="valid")
    # test_subgraphs  = pre_compute_subgraphs(args, g, df, mode='test' )

    ###################################################
    all_results = {
        "train_ap": [],
        "valid_ap": [],
        # 'test_ap' : [],
        "train_auc": [],
        "valid_auc": [],
        # 'test_auc' : [],
        "train_loss": [],
        "valid_loss": [],
        # 'test_loss': [],
    }

    low_loss = 100000
    user_train_total_time = 0
    user_epoch_num = 0
    best_epoch = -1
    best_auc = 0.0
    # 定义早停机制的参数
    patience = 10  # 允许验证集性能未提升的最大连续轮数
    counter = 0  # 记录验证集性能未提升的连续轮数

    if args.predict_class:
        num_classes = args.num_edgeType + 1
        train_AUROC = MulticlassAUROC(num_classes, average="macro", thresholds=None)
        valid_AUROC = MulticlassAUROC(num_classes, average="macro", thresholds=None)
        train_AUPRC = MulticlassAveragePrecision(
            num_classes, average="macro", thresholds=None
        )
        valid_AUPRC = MulticlassAveragePrecision(
            num_classes, average="macro", thresholds=None
        )
    else:
        train_AUROC = BinaryAUROC(thresholds=None)
        valid_AUROC = BinaryAUROC(thresholds=None)
        train_AUPRC = BinaryAveragePrecision(thresholds=None)
        valid_AUPRC = BinaryAveragePrecision(thresholds=None)

    for epoch in range(args.num_epoch):
        logging.info(f">>> Epoch {epoch + 1}")
        train_auc, train_ap, train_loss, time_train = run(
            model,
            optimizer,
            args,
            train_subgraphs,
            df,
            node_feats,
            edge_feats,
            train_AUROC,
            train_AUPRC,
            mode="train",
        )
        with torch.no_grad():
            valid_auc, valid_ap, valid_loss, time_valid = run(
                model,
                None,
                args,
                valid_subgraphs,
                df,
                node_feats,
                edge_feats,
                valid_AUROC,
                valid_AUPRC,
                mode="valid",
            )

        if valid_loss < low_loss:
            best_auc_state = copy.deepcopy(model.state_dict())
            best_auc = valid_auc
            low_loss = valid_loss
            best_epoch = epoch
            counter = 0  # 重置早停计数器
        else:
            counter += 1  # 验证集性能未提升，计数器加1

        user_train_total_time += time_train + time_valid
        user_epoch_num += 1

        # 检查早停条件
        if counter >= patience:
            logging.info(f"Early stopping at epoch {epoch + 1}")
            break

        all_results["train_ap"].append(train_ap)
        all_results["valid_ap"].append(valid_ap)

        all_results["valid_auc"].append(valid_auc)
        all_results["train_auc"].append(train_auc)

        all_results["train_loss"].append(train_loss)
        all_results["valid_loss"].append(valid_loss)

        _EIGEN_CACHE.clear()
        _SNAPSHOT_CACHE.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logging.info(f"best epoch {best_epoch}, auc score {best_auc}")
    return best_auc_state


def compute_sign_feats(node_feats, df, start_i, num_links, root_nodes, args):
    """
    计算 SIGN (Scalable Inception Graph Neural Networks) 特征。

    参数:
    node_feats (torch.Tensor): 输入的节点特征张量。
    df (pandas.DataFrame): 包含图边信息的 DataFrame，通常有 'src' 和 'dst' 列。
    start_i (int): 处理边信息时的起始索引。
    num_links (int): 链接的数量。
    root_nodes (list): 根节点的索引列表。
    args (argparse.Namespace): 包含配置参数的对象。

    返回:
    torch.Tensor: 计算得到的 SIGN 特征张量。
    """
    # 计算每个链接对应的根节点重复次数
    num_duplicate = len(root_nodes) // num_links
    # 获取图中节点的总数
    num_nodes = args.num_nodes

    # 生成从 0 到 len(root_nodes) - 1 的整数序列，并重塑为二维张量
    root_inds = torch.arange(len(root_nodes)).view(num_duplicate, -1)
    # 在第 1 维上分割张量为 1 个块，并将每个块展平为一维张量
    root_inds = [arr.flatten() for arr in root_inds.chunk(1, dim=1)]

    # 初始化输出特征张量，形状为 (len(root_nodes), node_feats.size(1))，并移动到指定设备
    output_feats = torch.zeros((len(root_nodes), node_feats.size(1)))
    # 初始化当前处理的边信息的索引
    i = start_i

    # 遍历每个根节点索引组
    for _root_ind in root_inds:

        # 如果是起始索引或者不需要进行结构跳数聚合，则直接复制原始节点特征
        if i == 0 or args.structure_hops == 0:
            sign_feats = node_feats.clone()
        else:
            # 计算边信息的起始索引，确保不小于 0
            prev_i = max(0, i - args.structure_time_gap)
            # 从 DataFrame 中截取相应范围的边信息
            cur_df = df[prev_i:i]  # 获取邻接矩阵的行和列索引（无向图）
            # 将源节点索引从 numpy 数组转换为 PyTorch 张量
            src = torch.from_numpy(cur_df.src.values)
            # 将目标节点索引从 numpy 数组转换为 PyTorch 张量
            dst = torch.from_numpy(cur_df.dst.values)
            # 构建无向图的边索引，将 src 和 dst 拼接两次
            edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
            # 对边索引进行去重，并返回去重后的边索引和每条边的出现次数
            edge_index, edge_cnt = torch.unique(edge_index, dim=1, return_counts=True)
            # 创建掩码，过滤自环边（源节点和目标节点相同的边）
            mask = edge_index[0] != edge_index[1]  # 忽略自环边
            # 构建稀疏邻接矩阵
            adj = SparseTensor(
                # 邻接矩阵中非零元素的值为 1
                value=torch.ones_like(edge_cnt[mask]).float(),
                # 邻接矩阵的行索引
                row=edge_index[0][mask].long(),
                # 邻接矩阵的列索引
                col=edge_index[1][mask].long(),
                # 邻接矩阵的形状
                sparse_sizes=(num_nodes, num_nodes),
            )
            # 对邻接矩阵进行行归一化，并移动到指定设备
            adj_norm = row_norm(adj)
            # 初始化 SIGN 特征列表，第一个元素为原始节点特征
            sign_feats = [node_feats]
            # 进行多跳邻域聚合
            for _ in range(args.structure_hops):
                # 通过矩阵乘法进行邻域聚合，并添加到 SIGN 特征列表
                sign_feats.append(adj_norm @ sign_feats[-1])
            # 将 SIGN 特征列表中的所有张量在第 0 维堆叠后求和
            sign_feats = torch.sum(torch.stack(sign_feats), dim=0)

        # 将计算得到的 SIGN 特征赋值给对应的根节点
        output_feats[_root_ind] = sign_feats[root_nodes[_root_ind]]

        # 更新当前处理的边信息的索引
        i += len(_root_ind) // num_duplicate

    # 返回计算得到的 SIGN 特征张量
    return output_feats.to(args.device)


def test(split_mode, model, args, metric, neg_sampler, g, df, node_feats, edge_feats):
    """Evaluate dynamic link prediction"""
    model.eval()
    logging.info(f"Starting {split_mode} phase...")

    # Pre-compute subgraphs
    test_subgraphs = pre_compute_subgraphs(
        args,
        g,
        df,
        mode="test" if split_mode == "test" else "valid",
        negative_sampler=neg_sampler,
        split_mode=split_mode,
    )

    # Get current dataframe based on split mode
    if split_mode == "test":
        cur_df = df[args.test_mask]
    elif split_mode == "val":
        cur_df = df[args.val_mask]

    neg_samples = 20
    cached_neg_samples = 20

    # Create test loader
    test_loader = cur_df.groupby(cur_df.index // args.batch_size)
    pbar = tqdm(total=len(test_loader))
    pbar.set_description(
        "%s mode with negative samples %d ..." % (split_mode, neg_samples)
    )

    # Initialize variables
    cur_inds = 0
    evaluator = Evaluator(name=args.dataset)
    perf_list = []

    # --- initialize AUC / AP metrics ---
    if args.predict_class:
        num_classes = args.num_edgeType + 1
        auc_metric = MulticlassAUROC(num_classes, average="macro", thresholds=None)
        ap_metric = MulticlassAveragePrecision(num_classes, average="macro", thresholds=None)
    else:
        auc_metric = BinaryAUROC(thresholds=None)
        ap_metric = BinaryAveragePrecision(thresholds=None)
    try:
        auc_metric.reset()
        ap_metric.reset()
    except Exception:
        pass

    logging.info(f"Starting prediction for {split_mode} set...")

    with torch.no_grad():
        for ind in range(len(test_loader)):
            # Get inputs for current batch
            inputs, subgraph_node_feats, cur_inds, structual_data = get_inputs_for_ind(
                test_subgraphs,
                "test" if split_mode == "test" else "tgb-val",
                cached_neg_samples,
                neg_samples,
                node_feats,
                edge_feats,
                cur_df,
                df,
                cur_inds,
                ind,
                args,
            )

            if args.use_riemannian_structure:
                loss, pred, edge_label = model(inputs, neg_samples, subgraph_node_feats,structual_data)
            else:
                loss, pred, edge_label = model(inputs, neg_samples, subgraph_node_feats)
            split = len(pred) // 2

            # 更新 AUC / AP
            try:
                auc_metric.update(pred, edge_label.to(torch.int))
                ap_metric.update(pred, edge_label.to(torch.int))
            except Exception:
                pass

            # Evaluate and store results
            perf_list.append(evaluate_mrr(pred, neg_samples))
            pbar.update(1)

            # Clear GPU cache periodically
            if ind % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Log progress
            if ind % 100 == 0:
                logging.info(f"Processed {ind} batches...")

    pbar.close()
    logging.info(f"Completed prediction for {split_mode} set.")

    # Calculate final metrics
    perf_mrr_mean = float(np.mean(perf_list))
    perf_mrr_std = float(np.std(perf_list))
    logging.info(
        f"{split_mode} results - {metric}: {perf_mrr_mean:.4f} ± {perf_mrr_std:.4f}"
    )

    # compute AUC/AUPRC
    try:
        total_auroc = float(auc_metric.compute())
    except Exception:
        total_auroc = None
    try:
        total_auprc = float(ap_metric.compute())
    except Exception:
        total_auprc = None

    logging.info(f"{split_mode} metrics - AUROC: {total_auroc}, AUPRC: {total_auprc}")

    # Clear memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 返回包含所有指标的元组： (mrr_mean, mrr_std, per_batch_list, auroc, auprc)
    return perf_mrr_mean, perf_mrr_std, perf_list, total_auroc, total_auprc



import torch
import numpy as np
from torch_geometric.data import Data, Batch
from torch_geometric.utils import get_laplacian
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
from collections import OrderedDict, deque

_EIGEN_CACHE = OrderedDict()
_EIGEN_CACHE_MAX_SIZE = 128
_EIGEN_FAST_EXIT_NODE_THRESH = 2000
_EIGEN_FAST_EXIT_EDGE_THRESH = 8000
_SNAPSHOT_CACHE = OrderedDict()
_SNAPSHOT_CACHE_MAX_SIZE = 64


def _make_cache_key(edge_index_cpu: torch.Tensor, num_nodes: int, embed_dim: int) -> tuple:
    edge_hash = hashlib.sha1(edge_index_cpu.numpy().tobytes()).hexdigest()
    return (num_nodes, embed_dim, edge_hash)


def _remember_eigvecs(cache_key: tuple, eigvecs: torch.Tensor):
    if eigvecs is None:
        return
    if eigvecs.device.type != "cpu":
        eigvecs = eigvecs.cpu()
    if len(_EIGEN_CACHE) >= _EIGEN_CACHE_MAX_SIZE:
        _EIGEN_CACHE.popitem(last=False)
    _EIGEN_CACHE[cache_key] = eigvecs.clone()


def _structure_based_embedding(
    num_nodes: int,
    embed_dim: int,
    device: torch.device,
    edge_index_cpu: torch.Tensor | None = None,
):
    methods = ("degree", "positional", "node_id", "random")
    for method in methods:
        try:
            if method == "degree" and edge_index_cpu is not None:
                deg = torch.bincount(edge_index_cpu[0], minlength=num_nodes).float()
                features = []
                for power in range(1, min(embed_dim, 4) + 1):
                    features.append((deg ** power).unsqueeze(-1))
                if features:
                    eigvecs = torch.cat(features, dim=-1)
                    if eigvecs.size(1) < embed_dim:
                        pad = torch.zeros(num_nodes, embed_dim - eigvecs.size(1))
                        eigvecs = torch.cat([eigvecs, pad], dim=-1)
                    return eigvecs.to(device)
            elif method == "positional":
                idx = torch.arange(num_nodes).float().unsqueeze(1)
                div_term = torch.exp(
                    torch.arange(0, embed_dim, 2).float()
                    * (-np.log(10000.0) / max(embed_dim - 1, 1))
                )
                pe = torch.zeros(num_nodes, embed_dim)
                pe[:, 0::2] = torch.sin(idx * div_term)
                pe[:, 1::2] = torch.cos(idx * div_term)
                return pe
            elif method == "node_id":
                emb = torch.zeros(num_nodes, embed_dim)
                cols = torch.arange(num_nodes) % embed_dim
                emb[torch.arange(num_nodes), cols] = 1.0
                return emb.to(device)
            else:
                return torch.randn(num_nodes, embed_dim, device=device) * 0.1
        except Exception:
            continue
    return torch.randn(num_nodes, embed_dim, device=device) * 0.1


def get_eigen_tokens_tensor(edge_index, num_nodes, embed_dim, device):
    """
    根据图结构计算拉普拉斯特征向量。(扩展策略版：全面的错误处理和多种备选方案)
    """
    edge_index_cpu = edge_index.cpu()
    cache_key = _make_cache_key(edge_index_cpu, num_nodes, embed_dim)
    cached = _EIGEN_CACHE.get(cache_key)
    if cached is not None:
        _EIGEN_CACHE.move_to_end(cache_key)
        return cached.to(device)
    edge_index = edge_index
    num_edges = edge_index_cpu.size(1)

    if (
        num_nodes > _EIGEN_FAST_EXIT_NODE_THRESH
        or num_edges > _EIGEN_FAST_EXIT_EDGE_THRESH
    ):
        eigvecs = _structure_based_embedding(
            num_nodes, embed_dim, device, edge_index_cpu=edge_index_cpu
        )
        try:
            _remember_eigvecs(cache_key, eigvecs)
        except Exception:
            pass
        return eigvecs.to(device)

    # 如果图太小或embed_dim太大，直接返回随机嵌入
    if num_nodes <= 2 or embed_dim >= num_nodes - 1:
        return torch.randn(num_nodes, embed_dim, device=device) * 0.1
    
    lap_edge_index, edge_weight = get_laplacian(edge_index, normalization="sym", num_nodes=num_nodes)
    row, col = lap_edge_index.cpu().numpy()
    L = csr_matrix((edge_weight.cpu().numpy(), (row, col)), shape=(num_nodes, num_nodes))
    
    # 检查拉普拉斯矩阵的条件数，如果太大说明数值不稳定
    try:
        # 添加小的正则化项来改善数值稳定性
        L = L + 1e-8 * csr_matrix(np.eye(num_nodes))
    except:
        pass
    
    k = min(embed_dim, num_nodes - 2)
    if k <= 0:
        return torch.zeros((num_nodes, embed_dim), device=device)
    
    # 大幅扩展的多级策略处理ARPACK错误
    strategies = [
        {'k': min(k, num_nodes // 3), 'ncv': min(3 * k + 15, num_nodes - 1), 'maxiter': num_nodes * 6, 'tol': 1e-6, 'which': 'SM'},
        {'k': min(k, num_nodes // 4), 'ncv': min(2 * k + 12, num_nodes - 1), 'maxiter': num_nodes * 8, 'tol': 1e-5, 'which': 'SM'},
        {'k': min(k // 2, num_nodes // 5), 'ncv': min(2 * k + 8, num_nodes - 1), 'maxiter': num_nodes * 10, 'tol': 1e-4, 'which': 'SM'},
        {'k': min(k // 2, 10), 'ncv': min(2 * k + 8, num_nodes - 1), 'maxiter': num_nodes * 12, 'tol': 1e-4, 'which': 'LM'},
        {'k': min(k // 3, 6), 'ncv': min(k + 6, num_nodes - 1), 'maxiter': num_nodes * 15, 'tol': 5e-4, 'which': 'SM'},
        {'k': min(4, num_nodes // 8), 'ncv': min(10, num_nodes - 1), 'maxiter': num_nodes * 20, 'tol': 1e-3, 'which': 'SM'},
    ]
    eigvecs = None
    max_strategy_attempts = 4
    for i, strategy in enumerate(strategies):
        actual_k = strategy['k']
        ncv = strategy['ncv']
        maxiter = strategy['maxiter']
        tol = strategy['tol']
        which = strategy.get('which', 'SM')
        
        # 严格的安全检查
        if actual_k <= 0 or actual_k >= num_nodes or ncv <= actual_k or ncv >= num_nodes:
            continue
            
        try:            
            # 尝试计算特征值和特征向量
            eigenvals, eigvecs_raw = eigs(
                L, 
                k=actual_k, 
                which=which, 
                ncv=ncv, 
                maxiter=maxiter,
                tol=tol
            )
            
            # 转换为PyTorch张量
            eigvecs = torch.from_numpy(eigvecs_raw.real).float()
            
            # 验证结果的有效性
            if torch.isnan(eigvecs).any() or torch.isinf(eigvecs).any():
                eigvecs = None
                continue
            break
            
        except (ArpackError, np.linalg.LinAlgError, ValueError) as e:
            eigvecs = None
            continue
        except Exception as e:
            eigvecs = None
            continue
        if i + 1 >= max_strategy_attempts and eigvecs is None:
            break
    
    # 如果所有ARPACK策略都失败了，尝试备选的数值方法
    if eigvecs is None:        
        # 备选方案1：使用scipy的其他特征值求解器
        alternative_methods = [
            # 使用稠密矩阵的标准特征值分解（适用于小图）
            {'method': 'dense_eigh', 'max_nodes': 1000},
            # 使用lobpcg方法（对于某些矩阵更稳定）
            {'method': 'lobpcg', 'max_nodes': 5000},
        ]
        
        for method_info in alternative_methods:
            if num_nodes > method_info['max_nodes']:
                continue
                
            try:
                if method_info['method'] == 'dense_eigh':
                    L_dense = L.toarray()
                    eigenvals, eigvecs_raw = np.linalg.eigh(L_dense)
                    # 取前k个最小的特征值对应的特征向量
                    k_actual = min(k, len(eigenvals) - 1)
                    eigvecs = torch.from_numpy(eigvecs_raw[:, :k_actual]).float()
                    break
                    
                elif method_info['method'] == 'lobpcg':
                    from scipy.sparse.linalg import lobpcg
                    k_actual = min(k, num_nodes // 4)
                    if k_actual > 0:
                        # LOBPCG需要初始猜测
                        X = np.random.rand(num_nodes, k_actual)
                        eigenvals, eigvecs_raw = lobpcg(L, X, largest=False, maxiter=maxiter)
                        eigvecs = torch.from_numpy(eigvecs_raw).float()
                        break
            except Exception as e:
                continue
    
    # 如果数值方法也失败了，使用图结构的备选方案
    if eigvecs is None:        
        structure_methods = [
            'degree_based',
            'random_walk_based', 
            'positional_encoding',
            'node_id_embedding',
            'random_initialization'
        ]
        
        for method in structure_methods:
            try:
                if method == 'degree_based':
                    # 基于节点度数的编码
                    degrees = np.array(L.sum(axis=1)).flatten()
                    degree_matrix = np.zeros((num_nodes, min(embed_dim, num_nodes)))
                    for i in range(min(embed_dim, num_nodes)):
                        degree_matrix[:, i] = np.power(degrees, i + 1)
                    # 归一化
                    degree_matrix = degree_matrix / (np.linalg.norm(degree_matrix, axis=0, keepdims=True) + 1e-8)
                    eigvecs = torch.from_numpy(degree_matrix).float()
                    
                elif method == 'random_walk_based':
                    # 基于随机游走的编码
                    P = L.copy()
                    P.data = 1.0 / (P.data + 1e-8)  # 转换为转移概率矩阵
                    rw_matrix = np.eye(num_nodes)
                    for step in range(min(embed_dim, 10)):
                        if step < embed_dim:
                            if step == 0:
                                encoding_matrix = rw_matrix.copy()
                            else:
                                encoding_matrix = np.column_stack([encoding_matrix, rw_matrix.sum(axis=1)])
                        rw_matrix = rw_matrix @ P.toarray()
                    eigvecs = torch.from_numpy(encoding_matrix[:, :embed_dim]).float()
                    
                elif method == 'positional_encoding':
                    # 位置编码
                    pos_encoding = torch.zeros(num_nodes, embed_dim)
                    for i in range(embed_dim):
                        for j in range(num_nodes):
                            if i % 2 == 0:
                                pos_encoding[j, i] = np.sin(j / (10000 ** (i / embed_dim)))
                            else:
                                pos_encoding[j, i] = np.cos(j / (10000 ** ((i-1) / embed_dim)))
                    eigvecs = pos_encoding
                    
                elif method == 'node_id_embedding':
                    # 节点ID嵌入
                    node_embedding = torch.zeros(num_nodes, embed_dim)
                    for i in range(num_nodes):
                        node_embedding[i, i % embed_dim] = 1.0
                    eigvecs = node_embedding
                    
                else:  # random_initialization
                    eigvecs = torch.randn(num_nodes, embed_dim) * 0.1
                
                if eigvecs is not None:
                    break
                    
            except Exception as e:
                continue
    
    # 最终的安全网：如果一切都失败了
    if eigvecs is None:
        eigvecs = torch.eye(num_nodes)[:, :min(embed_dim, num_nodes)]
        if eigvecs.shape[1] < embed_dim:
            padding = torch.zeros(num_nodes, embed_dim - eigvecs.shape[1])
            eigvecs = torch.cat([eigvecs, padding], dim=-1)
    
    # 调整维度以匹配embed_dim
    if eigvecs.shape[1] < embed_dim:
        # 如果特征向量数量不足，用零填充或重复最后一列
        if eigvecs.shape[1] > 0:
            # 重复最后一列直到达到embed_dim
            last_col = eigvecs[:, -1:].repeat(1, embed_dim - eigvecs.shape[1])
            eigvecs = torch.cat([eigvecs, last_col], dim=-1)
        else:
            padding = torch.zeros(eigvecs.shape[0], embed_dim)
            eigvecs = padding
    elif eigvecs.shape[1] > embed_dim:
        # 如果特征向量过多，截取前embed_dim个
        eigvecs = eigvecs[:, :embed_dim]
    
    # 最终的健全性检查和归一化
    if eigvecs.shape != (num_nodes, embed_dim):
        eigvecs = torch.randn(num_nodes, embed_dim) * 0.1
    
    # 添加归一化以提高数值稳定性
    eigvecs = eigvecs / (torch.norm(eigvecs, dim=1, keepdim=True) + 1e-8)
    eigvecs = torch.nan_to_num(eigvecs, nan=0.0, posinf=0.0, neginf=0.0)
    return eigvecs.to(device)


def _make_snapshot_cache_key(
    nodes: list,
    row: list,
    col: list,
    root_nodes: list,
    embed_dim: int,
    dataset_name: str,
) -> tuple:
    return (
        tuple(nodes),
        tuple(row),
        tuple(col),
        tuple(root_nodes),
        embed_dim,
        dataset_name,
    )


def _remember_snapshot(cache_key: tuple, snapshot_data: Data):
    if cache_key is None:
        return
    if len(_SNAPSHOT_CACHE) >= _SNAPSHOT_CACHE_MAX_SIZE:
        _SNAPSHOT_CACHE.popitem(last=False)
    _SNAPSHOT_CACHE[cache_key] = snapshot_data.clone().to("cpu")


def create_riemannian_data_snapshot(
    nodes: list, 
    row: list, 
    col: list, 
    root_nodes: list, 
    embed_dim: int, 
    device: torch.device,
    # 🆕 关键修改 1: 添加 dataset_name 参数
    dataset_name: str 
):
    """
    根据批次信息构建输入的Data对象。
    (最终版：增加了针对特定数据集的星型图采样逻辑)
    """
    
    cache_key = _make_snapshot_cache_key(
        nodes, row, col, root_nodes, embed_dim, dataset_name
    )
    cached_snapshot = _SNAPSHOT_CACHE.get(cache_key)
    if cached_snapshot is not None:
        _SNAPSHOT_CACHE.move_to_end(cache_key)
        return cached_snapshot.clone().to(device)
    
    # --- Part 1: (不变) 合并节点并创建新的映射与图结构 ---
    snapshot_global_nodes = sorted(list(set(nodes) | set(root_nodes)))
    snapshot_global_to_local_map = {global_id: i for i, global_id in enumerate(snapshot_global_nodes)}
    old_local_to_new_local_map = {old_idx: snapshot_global_to_local_map.get(global_id) for old_idx, global_id in enumerate(nodes)}
    
    num_snapshot_nodes = len(snapshot_global_nodes)
    if len(row) > 0 and len(col) > 0:
        valid_edges = [(old_local_to_new_local_map[r], old_local_to_new_local_map[c]) for r, c in zip(row, col) if old_local_to_new_local_map.get(r) is not None and old_local_to_new_local_map.get(c) is not None]
        if valid_edges:
            new_row, new_col = zip(*valid_edges)
            edge_index = torch.tensor([new_row, new_col], dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        
    snapshot_data = Data(num_nodes=num_snapshot_nodes, edge_index=edge_index)

    # --- Part 2: (不变) 计算拉普拉斯特征并创建 'tokens' 方法 ---
    eigvecs = get_eigen_tokens_tensor(snapshot_data.edge_index, snapshot_data.num_nodes, embed_dim, device)
    snapshot_data._eigvecs = eigvecs
    snapshot_data.tokens = lambda idx: snapshot_data._eigvecs[idx]
    snapshot_data.x = snapshot_data.tokens(torch.arange(snapshot_data.num_nodes))

    # --- Part 3: (不变) 采样默认结构词汇：BFS树 ---
    edge_index_cpu = snapshot_data.edge_index.cpu()
    adj, degrees = _build_undirected_adj(edge_index_cpu, snapshot_data.num_nodes)
    snapshot_data.batch_tree = _make_bfs_batch(adj, root_nodes, snapshot_data.num_nodes, device)

    # --- 🆕 关键修改 2: 针对性地采样新的结构词汇：星型图 ---
    if dataset_name in ["thgl-github-subset", "thgl-software-subset"]:
        snapshot_data.batch_star = _make_star_batch(
            adj,
            degrees,
            snapshot_data.num_nodes,
            device,
        )
    else:
        snapshot_data.batch_star = None
    
    # --- Part 5: (不变) 存储ID和掩码 ---
    snapshot_data.global_n_id = torch.tensor(snapshot_global_nodes, dtype=torch.long)
    root_nodes_local_indices = [snapshot_global_to_local_map.get(gid) for gid in root_nodes if gid in snapshot_global_to_local_map]
    snapshot_data.root_nodes_mask = torch.tensor(root_nodes_local_indices, dtype=torch.long)
    snapshot_data.n_id = torch.arange(snapshot_data.num_nodes)

    try:
        _remember_snapshot(cache_key, snapshot_data)
    except Exception:
        pass
    return snapshot_data.to(device)


def _build_undirected_adj(edge_index_cpu: torch.Tensor, num_nodes: int):
    adj = [[] for _ in range(num_nodes)]
    if edge_index_cpu.numel() == 0:
        degrees = torch.zeros(num_nodes, dtype=torch.long)
        return adj, degrees
    src = edge_index_cpu[0].tolist()
    dst = edge_index_cpu[1].tolist()
    for u, v in zip(src, dst):
        if 0 <= u < num_nodes and 0 <= v < num_nodes:
            adj[u].append(v)
            adj[v].append(u)
    degrees = torch.tensor([len(neigh) for neigh in adj], dtype=torch.long)
    return adj, degrees


def _build_bfs_tree(adj: list[list[int]], root: int):
    num_nodes = len(adj)
    visited = [False] * num_nodes
    queue = deque([root])
    visited[root] = True
    src_edges, dst_edges = [], []
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                queue.append(v)
                src_edges.append(u)
                dst_edges.append(v)
    return src_edges, dst_edges


def _make_bfs_batch(adj: list[list[int]], roots: list[int], num_nodes: int, device: torch.device):
    data_list = []
    unique_roots = sorted(set(r for r in roots if 0 <= r < num_nodes))
    if not unique_roots:
        unique_roots = list(range(min(num_nodes, 4)))  # 至多采样 4 个兜底
    for root in unique_roots:
        src, dst = _build_bfs_tree(adj, root)
        edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.empty((2, 0), dtype=torch.long)
        data_list.append(Data(edge_index=edge_index, num_nodes=num_nodes))
    return Batch.from_data_list(data_list).to(device)


def _make_star_batch(
    adj: list[list[int]],
    degrees: torch.Tensor,
    num_nodes: int,
    device: torch.device,
    hub_ratio: float = 0.1,
):
    if num_nodes == 0:
        empty = Data(edge_index=torch.empty((2, 0), dtype=torch.long), num_nodes=0)
        return Batch.from_data_list([empty]).to(device)
    num_hubs = max(1, int(num_nodes * hub_ratio))
    deg_vals, hub_idx = torch.sort(degrees, descending=True)
    hub_nodes = hub_idx[:num_hubs].tolist()
    data_list = []
    for hub in hub_nodes:
        neighbors = adj[hub]
        if not neighbors:
            continue
        src = torch.full((len(neighbors),), hub, dtype=torch.long)
        dst = torch.tensor(neighbors, dtype=torch.long)
        edge_index = torch.stack([src, dst], dim=0)
        data_list.append(Data(edge_index=edge_index, num_nodes=num_nodes))
    if not data_list:
        data_list = [Data(edge_index=torch.empty((2, 0), dtype=torch.long), num_nodes=num_nodes)]
    return Batch.from_data_list(data_list).to(device)