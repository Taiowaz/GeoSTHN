#!/bin/bash
set -e

ENV_NAME=${ENV_NAME:-hsact}

conda create -n "${ENV_NAME}" python=3.10 -y
conda run -n "${ENV_NAME}" pip install -e .

conda run -n "${ENV_NAME}" pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
conda run -n "${ENV_NAME}" pip install torch-sparse torch_geometric torch_scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

conda run -n "${ENV_NAME}" pip uninstall numpy -y
conda run -n "${ENV_NAME}" conda install numpy==1.23.5 pandas scikit-learn -y
conda run -n "${ENV_NAME}" pip install pytz clint torchmetrics geoopt

echo "Installation completed successfully!"
