import numpy as np
import logging


def load_model(args):
    motif_config_dict = {
        3: 13,
    }
    # get model
    if args.use_motif_feats:
        dim_in_node = args.node_feat_dims + motif_config_dict[args.motif_size]
    elif args.use_motif_metapath_feats:
        dim_in_node = args.node_feat_dims + 4
    else:
        dim_in_node = args.node_feat_dims
    edge_predictor_configs = {
        "dim_in_time": args.time_dims, 
        "dim_in_node": dim_in_node,  
        "predict_class": 1 if not args.predict_class else args.num_edgeType + 1,  
    }
    if args.model == "sthn":
        # !!!False!!!
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

    else:
        NotImplementedError()

    model = STHN_Interface(mixer_configs, edge_predictor_configs)
    for k, v in model.named_parameters():
        logging.info(f"{k}: {v.requires_grad}")

    logging_model_info(model)

    return model, args, link_pred_train


def logging_model_info(model):
    logging.info(model)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters])
    logging.info("Trainable Parameters: %d" % parameters)
