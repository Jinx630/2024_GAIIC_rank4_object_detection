# 初始化 Conda 环境
eval "$(conda shell.bash hook)"
conda create -n GAIIC python=3.8 -y
conda activate GAIIC
pip install torch==2.0.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install torchvision==0.15.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install ultralytics==8.2.18 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install transformers==4.35.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install ensemble-boxes==1.0.9 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -U openmim -i https://pypi.tuna.tsinghua.edu.cn/simple
mim install "mmcv>=2.0.0" -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install einops -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install IPython -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install Pillow==9.5.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install tensorboard -i https://pypi.tuna.tsinghua.edu.cn/simple