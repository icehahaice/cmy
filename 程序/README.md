
## Environment

由于BMNNSDK2中转换pytorch模型的工具链`bmnetP`当前对`torch==1.0.0`版本的模型支持较好，因此本代码基于pytorch 1.0.0

可按如下命令安装pytorch-cpu版本
```
pip3 install torch==1.0.0 torchvision==0.2.1
```

## 代码简介

#### 数据读取

```
load_data.py
```

#### 模型搭建

```
lenet5.py
```

#### 训练及测试

```
train.py
```

## 运行

在当前目录下执行

```commandline
python3 train.py
```

即可开始训练

执行
```commandline
python3 train.py
```
可对`0.jpg`进行预测，输出预测结果

## 转换成BModel
命令如下
```
python3 -m bmnetp --model=./model/lenet.zip --shapes=[1,1,32,32] --net_name="lenet" --opt=2 --outdir="./lenet" --target=BM1682
```