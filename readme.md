# CUB_200_2011鸟类图像分类

1. 数据集准备

    从网址https://data.caltech.edu/records/65de6-vp158中下载数据集，并解压到 dataset/文件夹下

2. 环境安装

    主要环境包括pytorch，timm，opencv-python, numpy, tensorboard

3. 训练

    执行python train_xxxx.py命令。 权重及tensorboard日志保存于 runs/ 文件夹下

4. 测试

    选择权重计算测试集准确率： python infer.py  --model "model_name" --ckpt "checkpoint_path"

    示例：

    python infer.py  --model resnet50 --ckpt runs/baseline/model.ckpt

5. tensorboard查看训练过程

    tensorboard --logdir=./runs