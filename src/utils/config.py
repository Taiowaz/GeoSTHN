import argparse
import os


def get_args(args=None):
    parser = argparse.ArgumentParser()
    # experiments
    parser.add_argument("--exper_name", type=str, default="thgl-forum-subset-debug")
    parser.add_argument("--exper_base_dir", type=str, default="exper")
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("--seed", type=int, help="Random seed", default=1)
    parser.add_argument(
        "--num_run", type=int, help="Number of iteration runs", default=10
    )

    # basic
    parser.add_argument("--dataset", type=str, default="thgl-forum-subset")
    parser.add_argument("--use_gpu", type=int, default=0, help="use gpu or not")
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="GPU device ID (-1 for all available GPUs)",
    )
    parser.add_argument("--batch_size", type=int, default=600)
    parser.add_argument("--max_edges", type=int, default=50)
    parser.add_argument("--num_edgeType", type=int, default=0, help="num of edgeType")
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--predict_class", action="store_true")

    # model
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--model", type=str, default="sthn")
    parser.add_argument("--neg_samples", type=int, default=1)
    parser.add_argument("--extra_neg_samples", type=int, default=5)
    parser.add_argument("--num_neighbors", type=int, default=50)
    parser.add_argument("--channel_expansion_factor", type=int, default=2)
    parser.add_argument("--sampled_num_hops", type=int, default=5)
    parser.add_argument("--time_dims", type=int, default=100)
    parser.add_argument("--hidden_dims", type=int, default=100)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--check_data_leakage", action="store_true")

    parser.add_argument("--ignore_node_feats", action="store_true")
    parser.add_argument("--node_feats_as_edge_feats", action="store_true")
    parser.add_argument("--ignore_edge_feats", action="store_true", default=False)
    parser.add_argument("--use_onehot_node_feats", action="store_true", default=False)
    parser.add_argument("--use_type_feats", action="store_true", default=True)

    parser.add_argument("--use_graph_structure", action="store_true", default=True)
    # 确定从历史边数据中提取多少时间范围内的图结构信息
    # train.compute_sign_feats中使用
    parser.add_argument("--structure_time_gap", type=int, default=2000)
    parser.add_argument("--structure_hops", type=int, default=1)

    parser.add_argument("--use_node_cls", action="store_true")
    parser.add_argument("--use_cached_subgraph", action="store_true", default=True)

    args = parser.parse_args(args)

    return args


