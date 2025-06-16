import timeit
import os
import sys
import pytz
from datetime import datetime

# 设置时区为中国时区
china_tz = pytz.timezone("Asia/Shanghai")

# 避免内存碎片化
# 当需要分配大于 128MB 的内存块时，尽量将请求拆分成更小的块（最大 128MB）
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# 添加项目根目录到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import logging
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
from tgb.utils.utils import set_random_seed, save_results
from utils.args_model import get_args
from utils.load_model import load_model
from utils.load_data import load_all_data
from utils.test import test
from utils.log import setup_logger


# 定义主函数
def main():
    # Start...
    start_overall = timeit.default_timer()
    # 1. 加载配置
    # ---------------------
    args = get_args()

    # 创建实验名称和输出目录
    timestamp = datetime.now(china_tz).strftime("%Y-%m-%d_%H-%M-%S")
    exp_name = args.exper_name
    # 存放实验相关文件
    exper_dir = os.path.join(args.exper_base_dir, exp_name)
    exper_time_dir = os.path.join(args.exper_base_dir, exp_name, timestamp)
    checkpoint_dir = os.path.join(exper_time_dir, "checkpoint")
    result_dir = os.path.join(exper_time_dir, "result")
    result_filename = f"{result_dir}/{args.model}_{args.dataset}_results.json"
    # 创建目录
    os.makedirs(exper_dir, exist_ok=True)
    os.makedirs(exper_time_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    # 2. 设置日志记录器
    # ---------------
    log_file = f"{exper_time_dir}/{exp_name}.log"
    setup_logger(log_file)
    logging.info(f"开始实验: {exp_name}")
    logging.info(f"配置信息: {args}")

    # 3. 设置设备 (CPU/GPU)
    # -------------------------
    if torch.cuda.is_available():
        if args.device == -1:  # 使用所有可用GPU
            args.device = torch.device("cuda")
            args.num_gpus = torch.cuda.device_count()
        else:  # 使用指定GPU
            args.device = torch.device(f"cuda:{args.device}")
            args.num_gpus = 1
    else:
        args.device = torch.device("cpu")
        args.num_gpus = 0
    logging.info(f"使用设备: {args.device}, GPU数量: {args.num_gpus}")

    # 4. 加载和预处理数据
    # ---------------------------
    logging.info("加载和预处理数据...")
    dataset = PyGLinkPropPredDataset(name=args.dataset, root="DATA")
    neg_sampler = dataset.negative_sampler
    data, node_feats, edge_feats, g, df, args = load_all_data(args, dataset)

    logging.info("数据加载和预处理完成。")

    # 5. 加载模型
    # -------------------
    logging.info("加载模型...")
    model, args, link_pred_train = load_model(args)
    if args.num_gpus > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(args.device)
    logging.info(f"模型结构: {model}")

    # 6. 定义损失函数和优化器
    # -------------------------------------
    metric = dataset.eval_metric

    # 7. 训练循环
    # ----------------
    logging.info("开始训练...")
    model.train()
    for run_idx in range(args.num_run):
        logging.info(
            "-------------------------------------------------------------------------------"
        )
        logging.info(f">>>>> Run: {run_idx} <<<<<")
        start_run = timeit.default_timer()

        # set the seed for deterministic results...
        torch.manual_seed(run_idx + args.seed)
        set_random_seed(run_idx + args.seed)

        # define an early stopper
        save_model_dir = checkpoint_dir
        save_model_id = f"{args.model}_{args.dataset}_{args.seed}_{run_idx}"
        # early_stopper = EarlyStopMonitor(save_model_dir=save_model_dir, save_model_id=save_model_id,
        #                                 tolerance=TOLERANCE, patience=PATIENCE)

        # ==================================================== Train & Validation
        # loading the validation negative samples

        # Link prediction
        start_val = timeit.default_timer()
        logging.info("Train link prediction task from scratch ...")

        model = link_pred_train(
            model.to(args.device), args, g, df, node_feats, edge_feats
        )

        # 保存训练好的模型
        if args.num_gpus > 1:
            torch.save(
                model.module.state_dict(),
                os.path.join(save_model_dir, f"{save_model_id}.pt"),
            )
        else:
            torch.save(
                model.state_dict(), os.path.join(save_model_dir, f"{save_model_id}.pt")
            )
        logging.info(
            f"模型已保存到: {os.path.join(save_model_dir, f'{save_model_id}.pt')}"
        )

        dataset.load_val_ns()

        # Validation ...
        perf_metrics_val_mean, perf_metrics_val_std, perf_list_val = test(
            "val",
            model.to(args.device),
            args,
            metric,
            neg_sampler,
            g,
            df,
            node_feats,
            edge_feats,
        )
        end_val = timeit.default_timer()

        logging.info(f"val: Evaluation Setting: >>> ONE-VS-MANY <<< ")
        logging.info(
            f"\tval: {metric}: {perf_metrics_val_mean: .4f} ± {perf_metrics_val_std: .4f}"
        )
        val_time = timeit.default_timer() - start_val
        logging.info(f"\tval: Elapsed Time (s): {val_time: .4f}")

        dataset.load_test_ns()
        # testing ...
        start_test = timeit.default_timer()
        perf_metrics_test_mean, perf_metrics_test_std, perf_list_test = test(
            "test",
            model.to(args.device),
            args,
            metric,
            neg_sampler,
            g,
            df,
            node_feats,
            edge_feats,
        )
        end_test = timeit.default_timer()

        logging.info(f"Test: Evaluation Setting: >>> ONE-VS-MANY <<< ")
        logging.info(
            f"\tTest: {metric}: {perf_metrics_test_mean: .4f} ± {perf_metrics_test_std: .4f}"
        )
        test_time = timeit.default_timer() - start_test
        logging.info(f"\tTest: Elapsed Time (s): {test_time: .4f}")

        save_results(
            {
                "model": args.model,
                "data": args.dataset,
                "run": run_idx,
                "seed": args.seed,
                f"val {metric}": f"{perf_metrics_val_mean: .4f} ± {perf_metrics_val_std: .4f}",
                f"test {metric}": f"{perf_metrics_test_mean: .4f} ± {perf_metrics_test_std: .4f}",
                "test_time": test_time,
                "tot_train_val_time": val_time,
            },
            result_filename,
        )

        logging.info(
            f">>>>> Run: {run_idx}, elapsed time: {timeit.default_timer() - start_run: .4f} <<<<<"
        )
        logging.info(
            "-------------------------------------------------------------------------------"
        )

    logging.info(
        f"Overall Elapsed Time (s): {timeit.default_timer() - start_overall: .4f}"
    )
    logging.info("==============================================================")


if __name__ == "__main__":
    main()
