from ast import arg
import numpy as np
import logging

from src.model.sthn import HeteroSTHN_Interface_rgfm_loss


def load_model(args):
    # get model
    edge_predictor_configs = {
        "dim_in_time": args.time_dims,  # 100
        "dim_in_node": args.node_feat_dims,  # 0
        "predict_class": 1 if not args.predict_class else args.num_edgeType + 1,  # 1
    }

    if args.model == "sthn":

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
        if args.use_riemannian_structure:
            from src.model.sthn import STHN_Interface_rgfm

            riemannian_configs = {
                "n_layers": args.rgfm_n_layers,
                "in_dim": args.rgfm_embed_dim,
                "embed_dim": args.rgfm_embed_dim,
                "hidden_dim": args.rgfm_hidden_dim,
                "dropout": args.rgfm_dropout,
                "bias": True,
                "activation": None,

                # "use_euclidean": True if args.use_euclidean==1 else False,
                # "use_hyperbolic": True if args.use_hyperbolic==1 else False,
                # "use_spherical": True if args.use_spherical==1 else False,
            }
            model = STHN_Interface_rgfm(
                mlp_mixer_configs=mixer_configs,
                edge_predictor_configs=edge_predictor_configs,
                riemannian_configs=riemannian_configs,
            )
        else:
            model = STHN_Interface(mixer_configs, edge_predictor_configs)

    elif args.model == "hetero_sthn":



        if args.predict_class:
            from src.model.sthn import (
                HeteroMulticlass_Interface as HeteroSTHN_Interface,
            )
        else:
            from src.model.sthn import HeteroSTHN_Interface, HeteroSTHN_Interface_rgfm
        from src.train_test import (
            link_pred_train,
        )


        edge_predictor_configs.update(
            {
                "edge_types": args.edge_types,
            }
        )


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
            "edge_types": args.edge_types,
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

                # "use_euclidean": True if args.use_euclidean==1 else False,
                # "use_hyperbolic": True if args.use_hyperbolic==1 else False,
                # "use_spherical": True if args.use_spherical==1 else False,
            }
            if args.use_ali_loss == 1:
                from src.model.sthn import (
                    HeteroSTHN_Interface_rgfm_loss as HeteroSTHN_Interface_rgfm,
                )

                model = HeteroSTHN_Interface_rgfm_loss(
                    mlp_mixer_configs=mixer_configs,
                    edge_predictor_configs=edge_predictor_configs,
                    edge_types=args.edge_types,
                    riemannian_configs=riemannian_configs,
                )
            else:
                model = HeteroSTHN_Interface_rgfm(
                    mlp_mixer_configs=mixer_configs,
                    edge_predictor_configs=edge_predictor_configs,
                    edge_types=args.edge_types,
                    riemannian_configs=riemannian_configs,
                )
        else:

            model = HeteroSTHN_Interface(
                mlp_mixer_configs=mixer_configs,
                edge_predictor_configs=edge_predictor_configs,
                edge_types=args.edge_types,
            )



    else:
        raise NotImplementedError(f"Model {args.model} not implemented")

    # for k, v in model.named_parameters():
    #     logging.info(f"{k}: {v.requires_grad}")

    # logging_model_info(model)

    return model, args, link_pred_train


def logging_model_info(model):
    logging.info(model)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters])
    logging.info("Trainable Parameters: %d" % parameters)
