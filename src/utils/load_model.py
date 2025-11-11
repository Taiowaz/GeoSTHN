from ast import arg
import numpy as np
import logging


def load_model(args):
    # get model
    edge_predictor_configs = {
        "dim_in_time": args.time_dims,  # 100
        "dim_in_node": args.node_feat_dims,  # 0
        "predict_class": 1 if not args.predict_class else args.num_edgeType + 1,  # 1
    }

    if args.model == "sthn":
        # 原始STHN模型 - 保持不变
        if args.predict_class:
            from src.model.sthn import Multiclass_Interface as STHN_Interface
        else:
            from src.model.sthn import STHN_Interface
        from src.train_test import link_pred_train

        mixer_configs = {
            "per_graph_size": args.max_edges,  # 50
            "time_channels": args.time_dims,  # 100
            "input_channels": args.edge_feat_dims,  # 14
            "hidden_channels": args.hidden_dims,  # 100
            "out_channels": args.hidden_dims,  # 100
            "num_layers": args.num_layers,  # 1
            "dropout": args.dropout,  # 0.1
            "channel_expansion_factor": args.channel_expansion_factor,  # 2
            "window_size": args.window_size,  # 5
            "use_single_layer": False,  # False
        }

        model = STHN_Interface(mixer_configs, edge_predictor_configs)

    elif args.model == "hetero_sthn":
        # 🆕 NEW: 异构STHN模型 - 使用我们设计的异构组件
        # 设置异构图默认参数

        if args.predict_class:
            from src.model.sthn import (
                HeteroMulticlass_Interface as HeteroSTHN_Interface,
            )
        else:
            from src.model.sthn import HeteroSTHN_Interface, HeteroSTHN_Interface_rgfm
        from src.train_test import (
            link_pred_train,
        )  # 🆕 NEW: 可以复用原有的训练函数！

        # 🆕 NEW: 异构边预测器配置（与原有配置兼容）
        edge_predictor_configs.update(
            {
                "edge_types": args.edge_types,  # 🆕 NEW: 添加边类型
            }
        )

        # 🆕 NEW: 异构mixer配置（与原有配置兼容）
        mixer_configs = {
            "per_graph_size": args.max_edges,  # 50
            "time_channels": args.time_dims,  # 100
            "input_channels": args.edge_feat_dims,  # 14
            "hidden_channels": args.hidden_dims,  # 100
            "out_channels": args.hidden_dims,  # 100
            "num_layers": args.num_layers,  # 1
            "dropout": args.dropout,  # 0.1
            "channel_expansion_factor": args.channel_expansion_factor,  # 2
            "window_size": args.window_size,  # 5
            "edge_types": args.edge_types,  # 🆕 NEW: 添加边类型
            "use_single_layer": False,  # False
        }

        if args.use_riemannian_structure:
            riemannian_configs = {
                "n_layers": args.rgfm_n_layers,
                "in_dim": args.rgfm_embed_dim,
                "embed_dim": args.rgfm_embed_dim,
                "hidden_dim": args.rgfm_hidden_dim,
                "dropout": args.rgfm_dropout,
                "bias": True,
                "activation": None,
            }
            model = HeteroSTHN_Interface_rgfm(
                mlp_mixer_configs=mixer_configs,
                edge_predictor_configs=edge_predictor_configs,
                edge_types=args.edge_types,  # 🆕 NEW: 传递边类型
                riemannian_configs=riemannian_configs,  # 🆕 NEW: 传递黎曼结构配置
            )
        else:
            # 🆕 NEW: 创建异构STHN模型（接口与原有模型几乎相同）
            model = HeteroSTHN_Interface(
                mlp_mixer_configs=mixer_configs,
                edge_predictor_configs=edge_predictor_configs,
                edge_types=args.edge_types,  # 🆕 NEW: 传递边类型
            )

        # 🆕 NEW: 可以复用原有的训练函数，因为我们保持了接口兼容性！
        # link_pred_train 函数可以不用修改

    else:
        raise NotImplementedError(f"Model {args.model} not implemented")

    for k, v in model.named_parameters():
        logging.info(f"{k}: {v.requires_grad}")

    logging_model_info(model)

    return model, args, link_pred_train


def logging_model_info(model):
    logging.info(model)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters])
    logging.info("Trainable Parameters: %d" % parameters)
