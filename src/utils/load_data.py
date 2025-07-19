import pandas as pd
import numpy as np
from tqdm import tqdm
import itertools
import torch
import logging

from src.utils.construct_subgraph import check_data_leakage


def load_graph(data):
    df = pd.DataFrame(
        {
            "idx": np.arange(len(data.t)),
            "src": data.src,
            "dst": data.dst,
            "time": data.t,
            "label": data.edge_type,
        }
    )

    num_nodes = max(int(df["src"].max()), int(df["dst"].max())) + 1

    ext_full_indptr = np.zeros(num_nodes + 1, dtype=np.int32)
    ext_full_indices = [[] for _ in range(num_nodes)]
    ext_full_ts = [[] for _ in range(num_nodes)]
    ext_full_eid = [[] for _ in range(num_nodes)]

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        src = int(row["src"])
        dst = int(row["dst"])

        ext_full_indices[src].append(dst)
        ext_full_ts[src].append(row["time"])
        ext_full_eid[src].append(idx)

    for i in tqdm(range(num_nodes)):
        ext_full_indptr[i + 1] = ext_full_indptr[i] + len(ext_full_indices[i])

    ext_full_indices = np.array(list(itertools.chain(*ext_full_indices)))
    ext_full_ts = np.array(list(itertools.chain(*ext_full_ts)))
    ext_full_eid = np.array(list(itertools.chain(*ext_full_eid)))

    logging.info("Sorting...")

    def tsort(i, indptr, indices, t, eid):
        beg = indptr[i]
        end = indptr[i + 1]
        sidx = np.argsort(t[beg:end])
        indices[beg:end] = indices[beg:end][sidx]
        t[beg:end] = t[beg:end][sidx]
        eid[beg:end] = eid[beg:end][sidx]

    for i in tqdm(range(ext_full_indptr.shape[0] - 1)):
        tsort(i, ext_full_indptr, ext_full_indices, ext_full_ts, ext_full_eid)

    logging.info("saving...")

    class NpzDict:
        def __init__(self, **arrays):
            self.arrays = arrays
        
        def __getitem__(self, key):
            return self.arrays[key]
    g = NpzDict(
        indptr=ext_full_indptr,
        indices=ext_full_indices,
        ts=ext_full_ts,
        eid=ext_full_eid
    )
    return g, df


def load_all_data(args, dataset):
    train_mask = dataset.train_mask
    val_mask = dataset.val_mask
    test_mask = dataset.test_mask
    data = dataset.get_TemporalData()

    # load graph
    g, df = load_graph(data)

    args.train_mask = train_mask.numpy()
    args.val_mask = val_mask.numpy()
    args.test_mask = test_mask.numpy()
    args.num_edges = len(df)

    logging.info(
        "Train %d, Valid %d, Test %d"
        % (sum(args.train_mask), sum(args.val_mask), sum(test_mask))
    )

    args.num_nodes = max(int(df["src"].max()), int(df["dst"].max())) + 1
    args.num_edges = len(df)

    logging.info("Num nodes %d, num edges %d" % (args.num_nodes, args.num_edges))

    # load feats
    node_feats, edge_feats = dataset.node_feat, dataset.edge_feat
    node_feat_dims = 0 if node_feats is None else node_feats.shape[1]
    edge_feat_dims = 0 if edge_feats is None else edge_feats.shape[1]

    # feature pre-processing
    # !!!False!!!
    if args.use_onehot_node_feats:
        logging.info(">>> Use one-hot node features")
        node_feats = torch.eye(args.num_nodes)
        node_feat_dims = node_feats.size(1)

    # !!!False!!!
    if args.ignore_node_feats:
        logging.info(">>> Ignore node features")
        node_feats = None
        node_feat_dims = 0

    # !!!True!!!
    if args.use_type_feats:
        edge_type = df.label.values
        logging.info(edge_type)
        logging.info(edge_type.sum())
        args.num_edgeType = len(set(edge_type.tolist()))
        edge_feats = torch.nn.functional.one_hot(
            torch.from_numpy(edge_type), num_classes=args.num_edgeType
        )
        edge_feat_dims = edge_feats.size(1)

    logging.info(
        "Node feature dim %d, edge feature dim %d" % (node_feat_dims, edge_feat_dims)
    )

    # double check (if data leakage then cannot continue the code)
    # !!!False!!!
    if args.check_data_leakage:
        check_data_leakage(args, g, df)

    args.node_feat_dims = node_feat_dims
    args.edge_feat_dims = edge_feat_dims

    if node_feats != None:
        node_feats = node_feats.to(args.device)
    if edge_feats != None:
        edge_feats = edge_feats.to(args.device)

    return data, node_feats, edge_feats, g, df, args
