# H-SACT

This repository contains the anonymous implementation for the temporal heterogeneous
graph link prediction experiments. The code is configured to run on the full THGL
datasets used in the paper.

## Repository Layout

```text
src/              Training, evaluation, and model entry points
riemanngfm/       Riemannian graph feature modules
tgb/              Dataset loaders and preprocessing utilities
scripts/          Reproduction and ablation launch scripts
```

## Environment

The setup script requires Conda and a C++ compiler with OpenMP support. It creates
a Python 3.10 environment, installs the local package in editable mode, and installs
PyTorch 2.1.0 with CUDA 12.1-compatible PyG wheels:

```bash
chmod +x env.sh
./env.sh
```

By default the script creates an environment named `hsact`. To use another name:

```bash
ENV_NAME=my_env ./env.sh
```

## Full Dataset Preparation

The experiments use the complete versions of the following datasets:

```text
thgl-forum
thgl-github
thgl-myket
thgl-software
```

Raw data and generated files are intentionally not included in the anonymous
archive. On first access, the loader prompts to download the selected full dataset.
To prepare files manually, place them under `tgb/DATA/` using these directory names:

```text
tgb/DATA/thgl_forum/
tgb/DATA/thgl_github/
tgb/DATA/thgl_myket/
tgb/DATA/thgl_software/
```

Each dataset directory should contain the full edge list, node type file, and
evaluation negative samples:

```text
<dataset-name>_edgelist.csv
<dataset-name>_nodetype.csv
<dataset-name>_val_ns.pkl
<dataset-name>_test_ns.pkl
```

For example, `tgb/DATA/thgl_forum/` should contain
`thgl-forum_edgelist.csv`, `thgl-forum_nodetype.csv`,
`thgl-forum_val_ns.pkl`, and `thgl-forum_test_ns.pkl`.

Processed `ml_*.pkl` cache files are created automatically on first use. If the
evaluation negative samples are not available, generate them after placing the raw
edge list and node type file:

```bash
python tgb/datasets/thgl_forum/thgl_forum_ns_gen.py
python tgb/datasets/thgl_github/thgl_github_ns_gen.py
python tgb/datasets/thgl_myket/thgl_myket_ns_gen.py
python tgb/datasets/thgl_software/thgl_software_ns_gen.py
```

## Run

Run commands from the repository root. The default configuration uses the CPU.
A single-run example is:

```bash
python src/main.py \
  --exper_name example \
  --dataset thgl-forum \
  --use_onehot_node_feats \
  --use_graph_structure \
  --model hetero_sthn \
  --use_riemannian_structure
```

To use a GPU, append the device selection:

```bash
python src/main.py \
  --exper_name example \
  --dataset thgl-forum \
  --use_onehot_node_feats \
  --use_graph_structure \
  --model hetero_sthn \
  --use_riemannian_structure \
  --use_gpu 1 \
  --device 0
```

Experiment outputs are written to `exper/<exper_name>/<dataset>/`.

## Reproduction Scripts

The scripts in `scripts/` launch reproduction and ablation jobs with `nohup`. Create
the log directory first. The launchers use the active Python interpreter by default:

```bash
mkdir -p run_log
bash scripts/hetero_sthn_rgfm.sh
```

You can override the interpreter or entry point without editing the scripts:

```bash
PYTHON=/path/to/python RUN_FILE=src/main.py bash scripts/hetero_sthn_rgfm.sh
```

Review the selected datasets and GPU device IDs inside a launcher before starting
it, because some launchers submit multiple jobs.

## Anonymous Submission

The repository is intended to be submitted without raw data, generated caches,
model checkpoints, experiment outputs, logs, local notebooks, or machine-specific
paths. The `.gitignore` file excludes these generated artifacts.

After committing the reviewed anonymous snapshot, export only Git-tracked files:

```bash
git archive --format=zip --output /tmp/anonymous-code.zip HEAD
```

Do not create the submission archive by compressing the working directory directly,
because local ignored data and experiment artifacts may still exist on your machine.
