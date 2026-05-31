import os
import pandas as pd
import torch
import numpy as np
import logging
from tqdm import tqdm
import time
import copy
from torch_sparse import SparseTensor
from torchmetrics.classification import MulticlassAUROC, MulticlassAveragePrecision
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from sklearn.preprocessing import MinMaxScaler
from src.utils.utils import evaluate_mrr, row_norm
from scipy.sparse.linalg import ArpackError
import networkx as nx
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_networkx, get_laplacian
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs

from src.utils.construct_subgraph import (
    get_random_inds,
    get_all_inds,
    construct_mini_batch_giant_graph,
    get_subgraph_sampler,
    pre_compute_subgraphs,
)


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
        structural_data = create_riemannian_data_snapshot(
            subgraph_data,
            args
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
    # scale
    scaler.fit(subgraph_edts.reshape(-1, 1))
    subgraph_edts = (
        scaler.transform(subgraph_edts.reshape(-1, 1)).ravel().astype(np.float32) * 1000
    )
    subgraph_edts = torch.from_numpy(subgraph_edts)

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
        if args.model =="hetero_sthn":
            inputs = [
                subgraph_edge_feats.to(args.device),
                subgraph_edts.to(args.device),
                len(has_temporal_neighbors),
                torch.tensor(all_inds).long(),
                torch.from_numpy(elabel[ind]).to(args.device),
            ]
        elif args.model == "sthn":
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
    return inputs, subgraph_node_feats, cur_inds, structural_data


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
        raise ("Use TGB evaluation")
        # model.eval()
        # cur_df = df[args.test_mask]
        # neg_samples = 1
        # cached_neg_samples = 1
        # cur_inds = args.val_edge_end

    train_loader = cur_df.groupby(cur_df.index // args.batch_size)
    pbar = tqdm(total=len(train_loader))
    pbar.set_description("%s mode with negative samples %d ..." % (mode, neg_samples))
    get_root_nodes(subgraphs, args, mode)

    loss_lst = []
    MLAUROC.reset()
    MLAUPRC.reset()

    hs = []
    for ind in range(len(train_loader)):
        inputs, subgraph_node_feats, cur_inds, structual_data = get_inputs_for_ind(
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
        if args.use_riemannian_structure:
            loss, pred, edge_label, h_batch = model(
                inputs, neg_samples, subgraph_node_feats, structual_data
            )
            hs.append(h_batch.detach().cpu().numpy())
            np.savez(f"{args.output_dir}/{args.dataset}_{mode}_hs.npz", hs=np.array(hs))
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

        pbar.update(1)

    
    pbar.close()
    total_auroc = MLAUROC.compute()
    total_auprc = MLAUPRC.compute()
    logging.info(
        f"{mode} mode with time {time_epoch:.4f}, AUROC {total_auroc:.4f}, AUPRC {total_auprc:.4f}, loss {loss.mean().item():.4f}"
    )
    return_loss = np.mean(loss_lst)
    return total_auroc, total_auprc, return_loss, time_epoch


def link_pred_train(model, args, g, df, node_feats, edge_feats):

    # optimizer = torch.optim.RMSprop(
    #     model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    # )
    optimizer = torch.optim.Adam(
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
    # 定义早停机制的参数
    patience = 5  # 允许验证集性能未提升的最大连续轮数
    counter = 0  # 记录验证集性能未提升的连续轮数

    # 【新增 1】用于记录总耗时和实际执行的 epoch 数量
    total_train_time = 0.0
    total_valid_time = 0.0
    actual_run_epochs = 0

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
                copy.deepcopy(model),
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
        # 【新增 2】累加耗时并更新实际运行轮数
        total_train_time += time_train
        total_valid_time += time_valid
        actual_run_epochs += 1

        if valid_loss < low_loss:
            best_auc_model = copy.deepcopy(model).cpu()
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

    logging.info(f"best epoch {best_epoch}, auc score {best_auc}")
    # 【新增 3】计算单轮平均耗时
    avg_train_time = total_train_time / actual_run_epochs
    avg_valid_time = total_valid_time / actual_run_epochs
    logging.info(f"best epoch {best_epoch}, auc score {best_auc}")
    logging.info(f"Avg Train Time per epoch: {avg_train_time:.4f} s")
    logging.info(f"Avg Valid(Prediction) Time per epoch: {avg_valid_time:.4f} s")
    return best_auc_model, avg_train_time, avg_valid_time


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
    get_root_nodes(test_subgraphs, args, "test")

    # Get current dataframe based on split mode
    if split_mode == "test":
        cur_df = df[args.test_mask]
    elif split_mode == "val":
        cur_df = df[args.val_mask]

    neg_samples = args.test_num_neg
    cached_neg_samples = args.test_num_neg

    # Create test loader
    test_loader = cur_df.groupby(cur_df.index // args.batch_size)
    pbar = tqdm(total=len(test_loader))
    pbar.set_description(
        "%s mode with negative samples %d ..." % (split_mode, neg_samples)
    )

    # Initialize variables
    cur_inds = 0
    perf_list = []
    all_preds = []
    all_labels = []

    if args.predict_class:
        num_classes = args.num_edgeType + 1
        auc_metric = MulticlassAUROC(num_classes, average="macro", thresholds=None)
        ap_metric = MulticlassAveragePrecision(
            num_classes, average="macro", thresholds=None
        )
    else:
        auc_metric = BinaryAUROC(thresholds=None)
        ap_metric = BinaryAveragePrecision(thresholds=None)

    auc_metric.reset()
    ap_metric.reset()

    logging.info(f"Starting prediction for {split_mode} set...")
    all_aucs = []
    all_aps = []
    with torch.no_grad():
        hs = []
        raw_temporal = []
        raw_structural = []
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
            
            raw_temporal.append(inputs[1].cpu().numpy())

            # Forward pass
            if args.use_riemannian_structure:
                raw_structural.append(structual_data.x.cpu().numpy())
                loss, pred, edge_label, h_batch = model(
                    inputs, neg_samples, subgraph_node_feats, structual_data
                )
            else:
                loss, pred, edge_label = model(inputs, neg_samples, subgraph_node_feats)
            print("pred.shape: ", pred.shape)
            split = len(pred) // 2
            hs.append(h_batch.detach().cpu().numpy())
            # mrr指标计算
            perf_list.append(evaluate_mrr(pred, neg_samples))
            pbar.update(1)

            # 计算AUC和AP
            num_src = len(edge_label) // (neg_samples + 1)
            for i in range(num_src):
                pred_batch_item = []
                label_batch_item = []
                for j in range(neg_samples + 1):
                    pred_batch_item.append(pred[i + j * num_src])
                    label_batch_item.append(edge_label[i + j * num_src])
                pred_batch_item = torch.stack(pred_batch_item)
                label_batch_item = torch.stack(label_batch_item)
                auc = auc_metric(pred_batch_item, label_batch_item.long())
                ap = ap_metric(pred_batch_item, label_batch_item.long())
                all_aucs.append(auc.item())
                all_aps.append(ap.item())
                all_preds.append(pred_batch_item)
                all_labels.append(label_batch_item)

            # Clear GPU cache periodically
            if ind % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Log progress
            if ind % 100 == 0:
                logging.info(f"Processed {ind} batches...")
        hs = np.array(hs)
        np.savez(f"{args.output_dir}/{args.dataset}_test_hs.npz", hs=hs)
        raw_temporal = np.array(raw_temporal)
        np.savez(
            f"{args.output_dir}/{args.dataset}_test_raw_temporal_feats.npz",
            raw_temporal=raw_temporal,
        )
        raw_structural = np.array(raw_structural)
        np.savez(
            f"{args.output_dir}/{args.dataset}_test_raw_structural_feats.npz",
            raw_structural=raw_structural,
        )
    pbar.close()
    logging.info(f"Completed prediction for {split_mode} set.")

    # mrr
    perf_metrics_mean = float(np.mean(perf_list))
    perf_metrics_std = float(np.std(perf_list))
    logging.info(
        f"{split_mode} results - {metric}: {perf_metrics_mean:.4f} ± {perf_metrics_std:.4f}"
    )
    # auc and ap
    auc = float(np.mean(all_aucs))
    auc_std = float(np.std(all_aucs))
    logging.info(f"{split_mode} results - auc: {auc:.4f} ± {auc_std:.4f}")

    ap = float(np.mean(all_aps))
    ap_std = float(np.std(all_aps))
    logging.info(f"{split_mode} results - ap: {ap:.4f} ± {ap_std:.4f}")

    all_preds = torch.stack(all_preds).cpu()
    all_labels = torch.stack(all_labels).cpu()
    save_path = os.path.join(args.output_dir, f"{args.dataset}_{split_mode}_preds.npz")
    np.savez(save_path, preds=all_preds.numpy(), labels=all_labels.numpy())

    logging.info(f"Saved predictions and labels to {save_path}")

    # Clear memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return (perf_metrics_mean, perf_metrics_std, perf_list, auc, ap)


def get_root_nodes(subgraphs, args, split_mode):
    subgraph_data = subgraphs[0]
    df_edges = pd.read_csv(
        f"tgb/DATA/{args.dataset.replace('-','_')}/{args.dataset}_edgelist.csv"
    )

    def cal_his_degree(node, node_time):
        # 修复了之前的代码中由于列名不一致导致的错误，现在根据是否存在 "timestamp" 列来正确地计算历史度数。
        if "timestamp" in df_edges.columns:
            his_df = df_edges[(df_edges["timestamp"] < node_time)]
            his_head = his_df[his_df["head"] == node]
            his_tail = his_df[his_df["tail"] == node]

        else:
            his_df = df_edges[(df_edges["ts"] < node_time)]
            his_head = his_df[his_df["src"] == node]
            his_tail = his_df[his_df["dst"] == node]

        return len(his_head) + len(his_tail)

    nodes = []
    degrees = []
    for subgra in subgraph_data:
        # print("len(subgra): ", len(subgra))
        nodes_dst_batch = []
        nodes_src_batch = []
        degrees_src_batch = []
        degrees_dst_batch = []
        for sub in subgra[: args.batch_size]:
            nodes_src_batch.append(sub["root_node"])
            node_t = sub["root_time"]
            his_deg = cal_his_degree(sub["root_node"], node_t)
            degrees_src_batch.append(his_deg)
        for sub in subgra[args.batch_size : 2 * args.batch_size]:
            nodes_dst_batch.append(sub["root_node"])
            node_t = sub["root_time"]
            his_deg = cal_his_degree(sub["root_node"], node_t)
            degrees_dst_batch.append(his_deg)
        nodes.append([nodes_src_batch, nodes_dst_batch])
        degrees.append([degrees_src_batch, degrees_dst_batch])
    nodes = np.array(nodes)
    np.savez(
        os.path.join(args.output_dir, f"{args.dataset}_{split_mode}_nodes.npz"),
        nodes=nodes,
    )
    degrees = np.array(degrees)
    np.savez(
        os.path.join(args.output_dir, f"{args.dataset}_{split_mode}_degrees.npz"),
        degrees=degrees,
    )


def get_eigen_tokens_tensor(edge_index, num_nodes, embed_dim, device):
    """
    根据图结构计算拉普拉斯特征向量。(扩展策略版：全面的错误处理和多种备选方案)
    """
    edge_index = edge_index.to(device)

    # 如果图太小或embed_dim太大，直接返回随机嵌入
    if num_nodes <= 2 or embed_dim >= num_nodes - 1:
        return torch.randn(num_nodes, embed_dim, device=device) * 0.1

    lap_edge_index, edge_weight = get_laplacian(
        edge_index, normalization="sym", num_nodes=num_nodes
    )
    row, col = lap_edge_index.cpu().numpy()
    L = csr_matrix(
        (edge_weight.cpu().numpy(), (row, col)), shape=(num_nodes, num_nodes)
    )

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
        # 第一组：标准SM策略，逐步降低要求
        {
            "k": min(k, num_nodes // 3),
            "ncv": min(3 * k + 15, num_nodes - 1),
            "maxiter": num_nodes * 8,
            "tol": 1e-6,
            "which": "SM",
        },
        {
            "k": min(k, num_nodes // 4),
            "ncv": min(2 * k + 12, num_nodes - 1),
            "maxiter": num_nodes * 10,
            "tol": 1e-5,
            "which": "SM",
        },
        {
            "k": min(k // 2, num_nodes // 5),
            "ncv": min(2 * k + 8, num_nodes - 1),
            "maxiter": num_nodes * 15,
            "tol": 1e-4,
            "which": "SM",
        },
        {
            "k": min(k // 3, num_nodes // 6),
            "ncv": min(k + 10, num_nodes - 1),
            "maxiter": num_nodes * 20,
            "tol": 1e-4,
            "which": "SM",
        },
        {
            "k": min(k // 4, 8),
            "ncv": min(k + 8, num_nodes - 1),
            "maxiter": num_nodes * 25,
            "tol": 1e-3,
            "which": "SM",
        },
        # 第二组：尝试LM策略（最大特征值）
        {
            "k": min(k // 2, 10),
            "ncv": min(2 * k + 8, num_nodes - 1),
            "maxiter": num_nodes * 12,
            "tol": 1e-5,
            "which": "LM",
        },
        {
            "k": min(k // 3, 8),
            "ncv": min(k + 8, num_nodes - 1),
            "maxiter": num_nodes * 18,
            "tol": 1e-4,
            "which": "LM",
        },
        {
            "k": min(k // 4, 6),
            "ncv": min(k + 6, num_nodes - 1),
            "maxiter": num_nodes * 25,
            "tol": 1e-3,
            "which": "LM",
        },
        # 第三组：尝试LR策略（实部最大）
        {
            "k": min(k // 3, 8),
            "ncv": min(k + 8, num_nodes - 1),
            "maxiter": num_nodes * 20,
            "tol": 1e-4,
            "which": "LR",
        },
        {
            "k": min(k // 4, 6),
            "ncv": min(k + 6, num_nodes - 1),
            "maxiter": num_nodes * 30,
            "tol": 1e-3,
            "which": "LR",
        },
        # 第四组：尝试SR策略（实部最小）
        {
            "k": min(k // 3, 8),
            "ncv": min(k + 8, num_nodes - 1),
            "maxiter": num_nodes * 20,
            "tol": 1e-4,
            "which": "SR",
        },
        {
            "k": min(k // 4, 6),
            "ncv": min(k + 6, num_nodes - 1),
            "maxiter": num_nodes * 30,
            "tol": 1e-3,
            "which": "SR",
        },
        # 第五组：最保守策略，k值非常小
        {
            "k": min(5, num_nodes // 8),
            "ncv": min(12, num_nodes - 1),
            "maxiter": num_nodes * 40,
            "tol": 1e-2,
            "which": "SM",
        },
        {
            "k": min(4, num_nodes // 10),
            "ncv": min(10, num_nodes - 1),
            "maxiter": num_nodes * 50,
            "tol": 1e-2,
            "which": "LM",
        },
        {
            "k": min(3, num_nodes // 12),
            "ncv": min(8, num_nodes - 1),
            "maxiter": num_nodes * 60,
            "tol": 1e-1,
            "which": "SM",
        },
        # 第六组：极端保守策略
        {
            "k": 2,
            "ncv": min(6, num_nodes - 1),
            "maxiter": num_nodes * 100,
            "tol": 1e-1,
            "which": "SM",
        },
        {
            "k": 1,
            "ncv": min(4, num_nodes - 1),
            "maxiter": num_nodes * 200,
            "tol": 1e-1,
            "which": "LM",
        },
    ]

    eigvecs = None

    for i, strategy in enumerate(strategies):
        actual_k = strategy["k"]
        ncv = strategy["ncv"]
        maxiter = strategy["maxiter"]
        tol = strategy["tol"]
        which = strategy.get("which", "SM")

        # 严格的安全检查
        if (
            actual_k <= 0
            or actual_k >= num_nodes
            or ncv <= actual_k
            or ncv >= num_nodes
        ):
            continue

        try:
            eigenvals, eigvecs_raw = eigs(
                L, k=actual_k, which=which, ncv=ncv, maxiter=maxiter, tol=tol
            )

            # 转换为PyTorch张量
            eigvecs = torch.from_numpy(eigvecs_raw.real).float().to(device)

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

    # 如果所有ARPACK策略都失败了，尝试备选的数值方法
    if eigvecs is None:

        # 备选方案1：使用scipy的其他特征值求解器
        alternative_methods = [
            # 使用稠密矩阵的标准特征值分解（适用于小图）
            {"method": "dense_eigh", "max_nodes": 1000},
            # 使用lobpcg方法（对于某些矩阵更稳定）
            {"method": "lobpcg", "max_nodes": 5000},
        ]

        for method_info in alternative_methods:
            if num_nodes > method_info["max_nodes"]:
                continue

            try:
                if method_info["method"] == "dense_eigh":
                    # print("Trying dense eigenvalue decomposition...")
                    L_dense = L.toarray()
                    eigenvals, eigvecs_raw = np.linalg.eigh(L_dense)
                    # 取前k个最小的特征值对应的特征向量
                    k_actual = min(k, len(eigenvals) - 1)
                    eigvecs = (
                        torch.from_numpy(eigvecs_raw[:, :k_actual]).float().to(device)
                    )
                    # print(f"Dense method succeeded with {k_actual} eigenvectors!")
                    break

                elif method_info["method"] == "lobpcg":
                    # print("Trying LOBPCG method...")
                    from scipy.sparse.linalg import lobpcg

                    k_actual = min(k, num_nodes // 4)
                    if k_actual > 0:
                        # LOBPCG需要初始猜测
                        X = np.random.rand(num_nodes, k_actual)
                        eigenvals, eigvecs_raw = lobpcg(
                            L, X, largest=False, maxiter=maxiter
                        )
                        eigvecs = torch.from_numpy(eigvecs_raw).float().to(device)
                        # print(f"LOBPCG method succeeded with {k_actual} eigenvectors!")
                        break

            except Exception as e:
                # print(f"Alternative method {method_info['method']} failed: {str(e)}")
                continue

    # 如果数值方法也失败了，使用图结构的备选方案
    if eigvecs is None:
        # print(
        #     "All numerical methods failed. Using graph-structure-based alternatives..."
        # )

        structure_methods = [
            "degree_based",
            "random_walk_based",
            "positional_encoding",
            "node_id_embedding",
            "random_initialization",
        ]

        for method in structure_methods:
            try:
                if method == "degree_based":
                    # print("Using degree-based encoding...")
                    # 基于节点度数的编码
                    degrees = np.array(L.sum(axis=1)).flatten()
                    degree_matrix = np.zeros((num_nodes, min(embed_dim, num_nodes)))
                    for i in range(min(embed_dim, num_nodes)):
                        degree_matrix[:, i] = np.power(degrees, i + 1)
                    # 归一化
                    degree_matrix = degree_matrix / (
                        np.linalg.norm(degree_matrix, axis=0, keepdims=True) + 1e-8
                    )
                    eigvecs = torch.from_numpy(degree_matrix).float().to(device)

                elif method == "random_walk_based":
                    # print("Using random walk based encoding...")
                    # 基于随机游走的编码
                    P = L.copy()
                    P.data = 1.0 / (P.data + 1e-8)  # 转换为转移概率矩阵
                    rw_matrix = np.eye(num_nodes)
                    for step in range(min(embed_dim, 10)):
                        if step < embed_dim:
                            if step == 0:
                                encoding_matrix = rw_matrix.copy()
                            else:
                                encoding_matrix = np.column_stack(
                                    [encoding_matrix, rw_matrix.sum(axis=1)]
                                )
                        rw_matrix = rw_matrix @ P.toarray()
                    eigvecs = (
                        torch.from_numpy(encoding_matrix[:, :embed_dim])
                        .float()
                        .to(device)
                    )

                elif method == "positional_encoding":
                    # print("Using positional encoding...")
                    # 位置编码
                    pos_encoding = torch.zeros(num_nodes, embed_dim, device=device)
                    for i in range(embed_dim):
                        for j in range(num_nodes):
                            if i % 2 == 0:
                                pos_encoding[j, i] = np.sin(
                                    j / (10000 ** (i / embed_dim))
                                )
                            else:
                                pos_encoding[j, i] = np.cos(
                                    j / (10000 ** ((i - 1) / embed_dim))
                                )
                    eigvecs = pos_encoding

                elif method == "node_id_embedding":
                    # print("Using node ID embedding...")
                    # 节点ID嵌入
                    node_embedding = torch.zeros(num_nodes, embed_dim, device=device)
                    for i in range(num_nodes):
                        node_embedding[i, i % embed_dim] = 1.0
                    eigvecs = node_embedding

                else:  # random_initialization
                    # print("Using random initialization...")
                    eigvecs = torch.randn(num_nodes, embed_dim, device=device) * 0.1

                if eigvecs is not None:
                    # print(f"Successfully generated embeddings using {method}")
                    break

            except Exception as e:
                # print(f"Structure method {method} failed: {str(e)}")
                continue

    # 最终的安全网：如果一切都失败了
    if eigvecs is None:
        # print("All methods failed. Using final fallback...")
        eigvecs = torch.eye(num_nodes, device=device)[:, : min(embed_dim, num_nodes)]
        if eigvecs.shape[1] < embed_dim:
            padding = torch.zeros(
                num_nodes, embed_dim - eigvecs.shape[1], device=device
            )
            eigvecs = torch.cat([eigvecs, padding], dim=-1)

    # 调整维度以匹配embed_dim
    if eigvecs.shape[1] < embed_dim:
        # 如果特征向量数量不足，用零填充或重复最后一列
        if eigvecs.shape[1] > 0:
            # 重复最后一列直到达到embed_dim
            last_col = eigvecs[:, -1:].repeat(1, embed_dim - eigvecs.shape[1])
            eigvecs = torch.cat([eigvecs, last_col], dim=-1)
        else:
            padding = torch.zeros(eigvecs.shape[0], embed_dim, device=device)
            eigvecs = padding
    elif eigvecs.shape[1] > embed_dim:
        # 如果特征向量过多，截取前embed_dim个
        eigvecs = eigvecs[:, :embed_dim]

    # 最终的健全性检查和归一化
    if eigvecs.shape != (num_nodes, embed_dim):
        # print(
        #     f"Warning: eigvecs shape {eigvecs.shape} doesn't match expected {(num_nodes, embed_dim)}"
        # )
        eigvecs = torch.randn(num_nodes, embed_dim, device=device) * 0.1

    # 添加归一化以提高数值稳定性
    eigvecs = eigvecs / (torch.norm(eigvecs, dim=1, keepdim=True) + 1e-8)

    return eigvecs

def create_riemannian_data_snapshot(
    subgraph_data, args
):
    """
    根据批次信息构建输入的Data对象。
    (最终版：增加了针对特定数据集的星型图采样逻辑)
    """
    nodes=subgraph_data["nodes"]
    row=subgraph_data["row"]
    col=subgraph_data["col"]
    root_nodes=subgraph_data["root_nodes"]
    embed_dim=args.rgfm_embed_dim
    device=args.device
    dataset_name=args.dataset
    # --- Part 1: (不变) 合并节点并创建新的映射与图结构 ---
    # 批次内所有节点的全局ID集合
    snapshot_global_nodes = sorted(list(set(nodes) | set(root_nodes)))
    # 全局ID到局部索引的映射, 
    snapshot_global_to_local_map = {
        global_id: i for i, global_id in enumerate(snapshot_global_nodes)
    }
    # 对于批次内nodes中的每个节点，找到其在新快照中的局部索引，构建映射
    old_local_to_new_local_map = {
        old_idx: snapshot_global_to_local_map.get(global_id)
        for old_idx, global_id in enumerate(nodes)
    }

    num_snapshot_nodes = len(snapshot_global_nodes)
    if len(row) > 0 and len(col) > 0:
        valid_edges = [
            (old_local_to_new_local_map[r], old_local_to_new_local_map[c])
            for r, c in zip(row, col)
            if old_local_to_new_local_map.get(r) is not None
            and old_local_to_new_local_map.get(c) is not None
        ]
        if valid_edges:
            new_row, new_col = zip(*valid_edges)
            edge_index = torch.tensor(
                [new_row, new_col], dtype=torch.long, device=device
            )
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

    snapshot_data = Data(num_nodes=num_snapshot_nodes, edge_index=edge_index)

    # --- Part 2: (不变) 计算拉普拉斯特征并创建 'tokens' 方法 ---
    # 为每个节点创建一个基于拉普拉斯特征的嵌入向量
    eigvecs = get_eigen_tokens_tensor(
        snapshot_data.edge_index, snapshot_data.num_nodes, embed_dim, device
    )
    snapshot_data._eigvecs = eigvecs
    # 定义一个方法，根据节点的局部索引返回其对应的拉普拉斯特征向量
    snapshot_data.tokens = lambda idx: snapshot_data._eigvecs[idx]
    snapshot_data.x = snapshot_data.tokens(
        torch.arange(snapshot_data.num_nodes, device=device)
    )

    # --- Part 3: (不变) 采样默认结构词汇：BFS树 ---
    G_undirected = to_networkx(snapshot_data, to_undirected=True)
    tree_list = []
    for i in range(snapshot_data.num_nodes):
        bfs_edges = list(nx.bfs_tree(G_undirected, i).edges())
        if not bfs_edges:
            tree_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        else:
            src, dst = zip(*bfs_edges)
            tree_edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)
        tree_list.append(
            Data(edge_index=tree_edge_index, num_nodes=snapshot_data.num_nodes)
        )
    snapshot_data.batch_tree = Batch.from_data_list(tree_list)

    # --- 🆕 关键修改 2: 针对性地采样新的结构词汇：星型图 ---
    if dataset_name in ["thgl-github", "thgl-software"]:
        star_list = []
        # 识别Hub节点 (例如，选择度数最高的前10%的节点)
        degrees = sorted(G_undirected.degree(), key=lambda x: x[1], reverse=True)
        num_hubs = max(1, int(snapshot_data.num_nodes * 0.1))  # 至少采样1个Hub
        hub_nodes = [node for node, degree in degrees[:num_hubs]]

        for hub_node in hub_nodes:
            neighbors = list(G_undirected.neighbors(hub_node))
            if not neighbors:
                continue

            src_star = [hub_node] * len(neighbors)
            dst_star = neighbors
            star_edge_index = torch.tensor(
                [src_star, dst_star], dtype=torch.long, device=device
            )
            star_list.append(
                Data(edge_index=star_edge_index, num_nodes=snapshot_data.num_nodes)
            )

        if star_list:
            snapshot_data.batch_star = Batch.from_data_list(star_list)
        else:
            # 创建一个空的Batch对象以保持数据结构一致性
            snapshot_data.batch_star = Batch.from_data_list(
                [
                    Data(
                        edge_index=torch.empty((2, 0), dtype=torch.long, device=device),
                        num_nodes=snapshot_data.num_nodes,
                    )
                ]
            )

    # --- Part 5: (不变) 存储ID和掩码 ---
    # 节点的全局ID记录
    snapshot_data.global_n_id = torch.tensor(
        snapshot_global_nodes, dtype=torch.long, device=device
    )
    # 根节点在新快照中的局部索引
    root_nodes_local_indices = [
        snapshot_global_to_local_map.get(gid)
        for gid in root_nodes
        if gid in snapshot_global_to_local_map
    ]
    # 根节点在新快照中的局部索引
    snapshot_data.root_nodes_mask = torch.tensor(
        root_nodes_local_indices, dtype=torch.long, device=device
    )
    # 图中非孤立点的局部索引,num_nodes有问题
    snapshot_data.n_id = torch.arange(snapshot_data.num_nodes, device=device)
    # 构建一个当前批次节点索引对应于全局节点ID的映射张量


    return snapshot_data.to(device)
