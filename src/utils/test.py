import torch
import numpy as np
from tqdm import tqdm
from src.utils.construct_subgraph import pre_compute_subgraphs
from utils.train import get_inputs_for_ind
from tgb.linkproppred.evaluate import Evaluator
import logging
from sklearn.metrics import roc_auc_score, average_precision_score


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

    logging.info(f"Starting prediction for {split_mode} set...")

    with torch.no_grad():
        for ind in range(len(test_loader)):
            # Get inputs for current batch
            inputs, subgraph_node_feats, cur_inds = get_inputs_for_ind(
                test_subgraphs,
                "test" if split_mode == "test" else "tgb-val",
                cached_neg_samples,
                neg_samples,
                node_feats,
                edge_feats,
                cur_df,
                cur_inds,
                ind,
                args,
            )

            # Forward pass
            loss, pred, edge_label = model(inputs, neg_samples, subgraph_node_feats)
            split = len(pred) // 2
            # Prepare evaluation input
            input_dict = {
                "y_pred_pos": np.array(pred.cpu()[:split].numpy()),
                "y_pred_neg": np.array(pred.cpu()[split:].numpy()),
                "eval_metric": [metric],
            }

            # Evaluate and store results
            perf_list.append(evaluator.eval(input_dict)[metric])
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
    perf_metrics_mean = float(np.mean(perf_list))
    perf_metrics_std = float(np.std(perf_list))
    logging.info(
        f"{split_mode} results - {metric}: {perf_metrics_mean:.4f} ± {perf_metrics_std:.4f}"
    )

    # Clear memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return perf_metrics_mean, perf_metrics_std, perf_list
