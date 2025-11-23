conda create -n geosthn python=3.10 -y
conda activate geosthn
pip install -e .
# 非5090
# pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
# pip install torch-sparse torch_geometric torch_scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
# 5090
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install torch-sparse torch_geometric torch_scatter -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
pip uninstall numpy
conda install numpy==1.23.5 -y
conda install pandas scikit-learn -y
pip install pytz clint torchmetrics geoopt