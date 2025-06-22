"""
Source: STHN: construct_subgraph.py
URL: https://github.com/celi52/STHN/blob/main/construct_subgraph.py

Notes: The NegLinkSampler is only used for STHN internal sampling and not for TGB
"""

from re import sub
import numpy as np
import logging


from tqdm import tqdm
from sampler_core import ParallelSampler
import os
import pickle


# get sampler
class NegLinkSampler:
    """
    From https://github.com/amazon-research/tgl/blob/main/sampler.py
    """

    def __init__(self, num_nodes):
        self.num_nodes = num_nodes

    def sample(self, n):
        return np.random.randint(self.num_nodes, size=n)


def get_parallel_sampler(g, num_neighbors=10):
    """
    Function wrapper of the C++ sampler (https://github.com/amazon-research/tgl/blob/main/sampler_core.cpp)
    Sample the 1-hop most recent neighbors of each node
    """

    configs = [
        g["indptr"],  # indptr --> fixed: data info
        g["indices"],  # indices --> fixed: data info
        g["eid"],  # eid --> fixed: data info
        g["ts"],  # ts --> fixed: data info
        32,  # num_thread_per_worker --> change this based on machine's setup
        1,  # num_workers --> change this based on machine's setup
        1,  # num_layers --> change this based on machine's setup
        [num_neighbors],  # num_neighbors --> hyper-parameters. Reddit 10, WIKI 30
        True,  # recent --> fixed: never touch
        False,  # prop_time --> never touch
        1,  # num_history --> fixed: never touch
        0,  # window_duration --> fixed: never touch
    ]

    sampler = ParallelSampler(*configs)
    neg_link_sampler = NegLinkSampler(g["indptr"].shape[0] - 1)
    return sampler, neg_link_sampler


##############################################################################
##############################################################################
##############################################################################
# sampling


def get_mini_batch(sampler, root_nodes, ts, num_hops):  # neg_samples is not used
    """
    Call function fetch_subgraph()
    Return: Subgraph of each node.
    """
    all_graphs = []

    for root_node, root_time in zip(root_nodes, ts):
        all_graphs.append(fetch_subgraph(sampler, root_node, root_time, num_hops))

    return all_graphs


def fetch_subgraph(sampler, root_node, root_time, num_hops):
    """
    Sample a subgraph for each node or node pair
    """
    all_row_col_times_nodes_eid = []

    # suppose sampling for both a single node and a node pair (two side of a link)
    if isinstance(root_node, list):
        nodes, ts = [i for i in root_node], [root_time for i in root_node]
    else:
        nodes, ts = [root_node], [root_time]

    # fetch all nodes+edges
    for _ in range(num_hops):
        sampler.sample(nodes, ts)
        ret = sampler.get_ret()  # 1-hop recent neighbors
        row, col, eid = ret[0].row(), ret[0].col(), ret[0].eid()
        nodes, ts = ret[0].nodes(), ret[0].ts().astype(np.float32)

        row_col_times_nodes_eid = np.stack(
            [ts[row], nodes[row], ts[col], nodes[col], eid]
        ).T
        all_row_col_times_nodes_eid.append(row_col_times_nodes_eid)
    all_row_col_times_nodes_eid = np.concatenate(all_row_col_times_nodes_eid, axis=0)

    # remove duplicate edges and sort according to the root node time (descending)
    all_row_col_times_nodes_eid = np.unique(all_row_col_times_nodes_eid, axis=0)[::-1]
    all_row_col_times_nodes = all_row_col_times_nodes_eid[:, :-1]
    eid = all_row_col_times_nodes_eid[:, -1]

    # remove duplicate (node+time) and sorted by time decending order
    all_row_col_times_nodes = np.array_split(all_row_col_times_nodes, 2, axis=1)
    times_nodes = np.concatenate(all_row_col_times_nodes, axis=0)
    times_nodes = np.unique(times_nodes, axis=0)[::-1]

    # each (node, time) pair identifies a node
    node_2_ind = dict()
    for ind, (time, node) in enumerate(times_nodes):
        node_2_ind[(time, node)] = ind

    # translate the nodes into new index
    row = np.zeros(len(eid), dtype=np.int32)
    col = np.zeros(len(eid), dtype=np.int32)
    for i, ((t1, n1), (t2, n2)) in enumerate(zip(*all_row_col_times_nodes)):
        row[i] = node_2_ind[(t1, n1)]
        col[i] = node_2_ind[(t2, n2)]

    # fetch get time + node information
    eid = eid.astype(np.int32)
    ts = times_nodes[:, 0].astype(np.float32)
    nodes = times_nodes[:, 1].astype(np.int32)
    dts = root_time - ts  # make sure the root node time is 0

    return {
        # edge info: sorted with descending row (src) node temporal order
        "row": row,  # src
        "col": col,  # dst
        "eid": eid,
        # node info
        "nodes": nodes,  # sorted by the ascending order of node's dts (root_node's dts = 0)
        "dts": dts,
        # graph info
        "num_nodes": len(nodes),
        "num_edges": len(eid),
        # root info
        "root_node": root_node,
        "root_time": root_time,
    }


def construct_mini_batch_giant_graph(all_graphs, max_num_edges):
    """
    Take the subgraph computed by fetch_subgraph() and combine it into a giant graph
    Return: the new indices of the graph
    将批量子图数据进行合并
    """

    all_rows, all_cols, all_eids, all_nodes, all_dts = [], [], [], [], []

    cumsum_edges = 0
    all_edge_indptr = [0]

    cumsum_nodes = 0
    all_node_indptr = [0]

    all_root_nodes = []
    all_root_times = []
    for all_graph in all_graphs:
        # record inds
        num_nodes = all_graph["num_nodes"]
        num_edges = min(all_graph["num_edges"], max_num_edges)

        # add graph information
        all_rows.append(all_graph["row"][:num_edges] + cumsum_nodes)
        all_cols.append(all_graph["col"][:num_edges] + cumsum_nodes)
        all_eids.append(all_graph["eid"][:num_edges])

        all_nodes.append(all_graph["nodes"])
        all_dts.append(all_graph["dts"])

        # update cumsum
        cumsum_nodes += num_nodes
        all_node_indptr.append(cumsum_nodes)

        cumsum_edges += num_edges
        all_edge_indptr.append(cumsum_edges)

        # add root nodes
        all_root_nodes.append(all_graph["root_node"])
        all_root_times.append(all_graph["root_time"])
    # for each edges
    all_rows = np.concatenate(all_rows).astype(np.int32)
    all_cols = np.concatenate(all_cols).astype(np.int32)
    all_eids = np.concatenate(all_eids).astype(np.int32)
    all_edge_indptr = np.array(all_edge_indptr).astype(np.int32)

    # for each nodes
    all_nodes = np.concatenate(all_nodes).astype(np.int32)
    all_dts = np.concatenate(all_dts).astype(np.float32)
    all_node_indptr = np.array(all_node_indptr).astype(np.int32)

    return {
        # for edges
        # 所有边的源节点索引数组，经过重新映射后的全局索引
        "row": all_rows,
        # 所有边的目标节点索引数组，经过重新映射后的全局索引
        "col": all_cols,
        # 所有边的原始ID数组，保持原始图中的边标识符
        "eid": all_eids,
        # 边时间差数组，计算为all_dts[all_cols] - all_dts[all_rows]，表示每条边两端节点的时间差
        "edts": all_dts[all_cols] - all_dts[all_rows],
        # 节点指针数组，标记每个子图在合并后图中的节点范围
        "all_node_indptr": all_node_indptr,
        # 边指针数组，标记每个子图在合并后图中的边范围
        "all_edge_indptr": all_edge_indptr,
        # for nodes
        # 所有节点的原始ID数组，来自各个子图的节点
        "nodes": all_nodes,
        # 所有节点的时间差数组，相对于各自根节点的时间差
        "dts": all_dts,
        # general information
        # 合并后图的总节点数
        "all_num_nodes": cumsum_nodes,
        # 合并后图的总边数
        "all_num_edges": cumsum_edges,
        # root nodes
        #  所有子图的根节点ID数组
        "root_nodes": np.array(all_root_nodes, dtype=np.int32),
        # 所有子图的根节点时间戳数组
        "root_times": np.array(all_root_times, dtype=np.float32),
    }


##############################################################################
##############################################################################
##############################################################################


def logging_subgraph_data(subgraph_data):
    """
    Used to double check see if the sampled graph is as expected
    """
    for key, vals in subgraph_data.items():
        if isinstance(vals, np.ndarray):
            logging.info(f"key:{key};vals{vals.shape}")
        else:
            logging.info(f"key:{key};vals{vals}")


class SubgraphSampler:
    def __init__(self, all_root_nodes, all_ts, sampler, args):
        self.all_root_nodes = all_root_nodes
        self.all_ts = all_ts
        self.sampler = sampler
        self.sampled_num_hops = args.sampled_num_hops

    def mini_batch(self, ind, mini_batch_inds):
        root_nodes = self.all_root_nodes[ind][mini_batch_inds]
        ts = self.all_ts[ind][mini_batch_inds]
        return get_mini_batch(self.sampler, root_nodes, ts, self.sampled_num_hops)


def get_subgraph_sampler(args, g, df, mode):
    ###################################################
    # get cached file_name
    if mode == "train":
        extra_neg_samples = args.extra_neg_samples
    else:
        extra_neg_samples = 1

    ###################################################
    # for each node, sample its neighbors with the most recent neighbors (sorted)
    logging.info(f"Sample subgraphs ... for {mode} mode")
    sampler, neg_link_sampler = get_parallel_sampler(g, args.num_neighbors)

    ###################################################
    # setup modes
    if mode == "train":
        cur_df = df[args.train_mask]

    elif mode == "valid":
        cur_df = df[args.val_mask]

    elif mode == "test":
        cur_df = df[args.test_mask]

    loader = cur_df.groupby(cur_df.index // args.batch_size)
    logging.info(
        f"Current index: {cur_df.index}, Batch index: {cur_df.index // args.batch_size}"
    )
    pbar = tqdm(total=len(loader))
    pbar.set_description(
        "Pre-sampling: %s mode with negative sampleds %s ..."
        % (mode, extra_neg_samples)
    )

    all_root_nodes = []
    all_ts = []
    for _, rows in loader:

        root_nodes = np.concatenate(
            [
                rows.src.values,
                rows.dst.values,
                neg_link_sampler.sample(len(rows) * extra_neg_samples),
            ]
        ).astype(np.int32)
        all_root_nodes.append(root_nodes)

        # time-stamp for node = edge time-stamp
        ts = np.tile(rows.time.values, extra_neg_samples + 2).astype(np.float32)
        all_ts.append(ts)

        pbar.update(1)
    pbar.close()
    return SubgraphSampler(all_root_nodes, all_ts, sampler, args)


######################################################################################################
######################################################################################################
######################################################################################################
# for small dataset, we can cache each graph
def pre_compute_subgraphs(
    args, g, df, mode, negative_sampler=None, split_mode="test", cache=True
):
    """
    预计算子图并在可能的情况下缓存结果。

    参数:
    args (Namespace): 包含各种配置参数的命名空间对象。
    g (dict): 图数据，包含邻接表、边信息等。
    df (DataFrame): 包含图的边信息的 Pandas 数据框。
    mode (str): 模式，可选值为 "train", "valid", "test"。
    negative_sampler (object, 可选): 负样本采样器，默认为 None。
    split_mode (str, 可选): 分割模式，默认为 "test"。
    cache (bool, 可选): 是否缓存预计算的子图，默认为 True。

    返回:
    tuple: 包含预计算的子图列表和对应的边标签列表的元组。
    (all_subgraphs, all_elabel)
    all_subgraphs: [[batch_size*extra_neg_samples], (len(loader)//batch_size)]
    all_elabel: [[batch_size],(len(loader)//batch_size)]
    """
    ###################################################
    # 获取缓存文件名
    if mode == "train":
        # 训练模式使用额外的负样本数量
        extra_neg_samples = args.extra_neg_samples
    else:
        # 验证和测试模式使用 1 个负样本
        extra_neg_samples = 1

    # 构建缓存文件的路径
    base_dir = os.path.join(os.getcwd(), "tgb/DATA", args.dataset.replace("-", "_"))
    args_path = f"{mode}_neg_sample_neg{extra_neg_samples}_bs{args.batch_size}_hops{args.sampled_num_hops}_neighbors{args.num_neighbors}"
    fn_subgraph_data = os.path.join(base_dir, f"{args_path}.pickle")
    fn_subgraph_llm_encode = os.path.join(base_dir, f"{args_path}_llm_encode.npy")
    ###################################################

    # 检查缓存文件是否存在
    if os.path.exists(fn_subgraph_data):
        # 若存在，则加载缓存的子图和边标签
        subgraph_elabel = pickle.load(open(fn_subgraph_data, "rb"))
        subgraph_llm_encode = np.load(fn_subgraph_llm_encode)
        logging.info(f"Successfully load subgraphs from {fn_subgraph_data}")
        logging.info(
            f"Successfully load subgraphs llm encode from {fn_subgraph_llm_encode}"
        )

    else:
        ##################################################
        # 为每个节点采样最近邻节点（按时间排序）
        logging.info(f"Sample subgraphs ... for {mode} mode")
        # 获取并行采样器和负链接采样器
        sampler, neg_link_sampler = get_parallel_sampler(g, args.num_neighbors)

        ###################################################
        # 根据不同模式设置当前数据框
        if mode == "train":
            cur_df = df[args.train_mask]

        elif mode == "valid":
            cur_df = df[args.val_mask]

        elif mode == "test":
            cur_df = df[args.test_mask]

        # 按批次大小对数据框进行分组
        loader = cur_df.groupby(cur_df.index // args.batch_size)
        # 初始化进度条
        pbar = tqdm(total=len(loader))
        pbar.set_description("Pre-sampling: %s mode" % (mode,))

        ###################################################
        # 初始化存储所有子图和边标签的列表
        all_subgraphs = []
        all_elabel = []
        # 重置采样器
        sampler.reset()
        for _, rows in loader:
            # 如果提供了负样本采样器，则使用它进行负样本采样
            if negative_sampler is not None:
                neg_batch_list = negative_sampler.query_batch(
                    rows.src.values,
                    rows.dst.values,
                    rows.time.values,
                    rows.label.values,
                    split_mode=split_mode,
                )
                neg_batch_list = np.concatenate(neg_batch_list)
                # 计算额外负样本的数量
                extra_neg_samples = neg_batch_list.shape[0] // len(rows)
            else:
                # 否则使用负链接采样器进行采样
                neg_batch_list = neg_link_sampler.sample(len(rows) * extra_neg_samples)

            # 拼接源节点、目标节点和负样本节点
            root_nodes = np.concatenate(
                [rows.src.values, rows.dst.values, neg_batch_list]
            ).astype(np.int32)

            # 节点的时间戳等于边的时间戳
            ts = np.tile(rows.time.values, extra_neg_samples + 2).astype(np.float32)
            # 记录当前批次的边标签
            all_elabel.append(rows.label.values)
            # 为根节点采样子图并添加到列表中
            all_subgraphs.append(
                get_mini_batch(sampler, root_nodes, ts, args.sampled_num_hops)
            )

            # 更新进度条
            pbar.update(1)
        # 关闭进度条
        pbar.close()
        # 将子图列表和边标签列表打包成元组
        subgraph_elabel = (all_subgraphs, all_elabel)

        if cache:
            try:
                # 将子图和边标签缓存到文件中
                pickle.dump(
                    subgraph_elabel,
                    open(fn_subgraph_data, "wb"),
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
                logging.info(f"Successfully cached subgraphs to {fn_subgraph_data}")
            except Exception as e:
                # 记录缓存失败的错误信息
                logging.error(f"Failed to cache subgraphs: {e}")
                # 根据需求决定是否继续执行
                # raise  # 如果缓存失败需要中断，可以取消注释

        ##################################################

    return subgraph_elabel, subgraph_llm_encode


def get_random_inds(num_subgraph, cached_neg_samples, neg_samples):
    ###################################################
    batch_size = num_subgraph // (2 + cached_neg_samples)
    pos_src_inds = np.arange(batch_size)
    pos_dst_inds = np.arange(batch_size) + batch_size
    # 0与1索引为源节点与目的节点的索引，因此，从后面的负样本索引中随机选取一个
    neg_dst_inds = np.random.randint(
        low=2, high=2 + cached_neg_samples, size=batch_size * neg_samples
    )
    neg_dst_inds = batch_size * neg_dst_inds + np.arange(batch_size)
    mini_batch_inds = np.concatenate([pos_src_inds, pos_dst_inds, neg_dst_inds]).astype(
        np.int32
    )
    ###################################################

    return mini_batch_inds


def get_all_inds(num_subgraph, neg_samples):
    ###################################################
    batch_size = num_subgraph // (2 + neg_samples)
    pos_src_inds = np.arange(batch_size)
    pos_dst_inds = np.arange(batch_size) + batch_size
    neg_dst_inds = batch_size * 2 + np.arange(batch_size * neg_samples)
    mini_batch_inds = np.concatenate([pos_src_inds, pos_dst_inds, neg_dst_inds]).astype(
        np.int32
    )
    ###################################################

    return mini_batch_inds


def check_data_leakage(args, g, df):
    """
    This is a function to double if the sampled graph has eid greater than the positive node pairs eid (if no then no data leakage)
    """
    for mode in ["train", "valid", "test"]:

        if mode == "train":
            cur_df = df[: args.train_edge_end]
        elif mode == "valid":
            cur_df = df[args.train_edge_end : args.val_edge_end]
        elif mode == "test":
            cur_df = df[args.val_edge_end :]

        loader = cur_df.groupby(cur_df.index // args.batch_size)
        subgraphs = pre_compute_subgraphs(args, g, df, mode)

        for i, (_, rows) in enumerate(loader):
            root_nodes = np.concatenate([rows.src.values, rows.dst.values]).astype(
                np.int32
            )
            eids = np.tile(rows.index.values, 2)
            cur_subgraphs = subgraphs[i][: args.batch_size * 2]

            for eid, cur_subgraph in zip(eids, cur_subgraphs):
                all_eids_in_subgraph = cur_subgraph["eid"]
                if len(all_eids_in_subgraph) == 0:
                    continue
                # all edges in the sampled graph has eid smaller than the target edge's eid, i.e,. sampled links never seen before
                assert sum(all_eids_in_subgraph < eid) == len(all_eids_in_subgraph)

    logging.info("Does not detect information leakage ...")
