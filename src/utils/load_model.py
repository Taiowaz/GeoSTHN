import numpy as np
import logging


def load_model(args):
    # get model
    edge_predictor_configs = {
        "dim_in_time": args.time_dims,
        "dim_in_node": args.node_feat_dims,
        "predict_class": 1 if not args.predict_class else args.num_edgeType + 1,
    }
    if args.model == "sthn":
        # !!!False!!!
        if args.predict_class:
            from src.model.sthn import Multiclass_Interface as STHN_Interface
        else:
            from src.model.sthn import STHN_Interface
        from src.utils.link_pred_train_utils import link_pred_train

        mixer_configs = {
            "per_graph_size": args.max_edges,
            "time_channels": args.time_dims,
            "input_channels": args.edge_feat_dims,
            "hidden_channels": args.hidden_dims,
            "out_channels": args.hidden_dims,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "channel_expansion_factor": args.channel_expansion_factor,
            "window_size": args.window_size,
            "use_single_layer": False,
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
