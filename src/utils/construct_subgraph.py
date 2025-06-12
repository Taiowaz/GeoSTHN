"""
Source: STHN: construct_subgraph.py
URL: https://github.com/celi52/STHN/blob/main/construct_subgraph.py

Notes: The NegLinkSampler is only used for STHN internal sampling and not for TGB
"""

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
        "row": all_rows,
        "col": all_cols,
        "eid": all_eids,
        "edts": all_dts[all_cols] - all_dts[all_rows],
        # number of subgraphs + 1
        "all_node_indptr": all_node_indptr,
        "all_edge_indptr": all_edge_indptr,
        # for nodes
        "nodes": all_nodes,
        "dts": all_dts,
        # general information
        "all_num_nodes": cumsum_nodes,
        "all_num_edges": cumsum_edges,
        # root nodes
        "root_nodes": np.array(all_root_nodes, dtype=np.int32),
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
    ###################################################
    # get cached file_name
    if mode == "train":
        extra_neg_samples = args.extra_neg_samples
    else:
        extra_neg_samples = 1

    fn = os.path.join(
        os.getcwd(),
        "tgb/DATA",
        args.dataset.replace("-", "_"),
        "%s_neg_sample_neg%d_bs%d_hops%d_neighbors%d.pickle"
        % (
            mode,
            extra_neg_samples,
            args.batch_size,
            args.sampled_num_hops,
            args.num_neighbors,
        ),
    )
    ###################################################

    # # try:
    if os.path.exists(fn):
        subgraph_elabel = pickle.load(open(fn, "rb"))
        logging.info(f"Successfully load subgraphs to {fn}")

    else:
        ##################################################
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
        pbar = tqdm(total=len(loader))
        pbar.set_description("Pre-sampling: %s mode" % (mode,))

        ###################################################
        all_subgraphs = []
        all_elabel = []
        sampler.reset()
        for _, rows in loader:

            if negative_sampler is not None:
                neg_batch_list = negative_sampler.query_batch(
                    rows.src.values,
                    rows.dst.values,
                    rows.time.values,
                    rows.label.values,
                    split_mode=split_mode,
                )
                neg_batch_list = np.concatenate(neg_batch_list)
                extra_neg_samples = neg_batch_list.shape[0] // len(rows)
            else:
                neg_batch_list = neg_link_sampler.sample(len(rows) * extra_neg_samples)

            root_nodes = np.concatenate(
                [rows.src.values, rows.dst.values, neg_batch_list]
            ).astype(np.int32)

            # time-stamp for node = edge time-stamp
            ts = np.tile(rows.time.values, extra_neg_samples + 2).astype(np.float32)
            all_elabel.append(rows.label.values)
            all_subgraphs.append(
                get_mini_batch(sampler, root_nodes, ts, args.sampled_num_hops)
            )

            pbar.update(1)
        pbar.close()
        subgraph_elabel = (all_subgraphs, all_elabel)

        if cache:
            try:
                pickle.dump(
                    subgraph_elabel, open(fn, "wb"), protocol=pickle.HIGHEST_PROTOCOL
                )
                logging.info(f"Successfully cached subgraphs to {fn}")
            except Exception as e:
                logging.error(f"Failed to cache subgraphs: {e}")
                # 根据需求决定是否继续执行
                # raise  # 如果缓存失败需要中断，可以取消注释

        ##################################################

    return subgraph_elabel


def get_random_inds(num_subgraph, cached_neg_samples, neg_samples):
    ###################################################
    batch_size = num_subgraph // (2 + cached_neg_samples)
    pos_src_inds = np.arange(batch_size)
    pos_dst_inds = np.arange(batch_size) + batch_size
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