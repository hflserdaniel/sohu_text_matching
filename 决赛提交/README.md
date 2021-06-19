### sohu_matching

#### 小组：**分比我们低的都是帅哥**

#### 决赛Docker运行说明

本项目的Docker构建过程符合提交指南要求，运行官方给出的测试命令即可进行推断：

```bash
docker run --rm -it --gpus all \
  -v ${TestInputDir}:/data/input \
  -v ${TestOutputDir}:/data/output \
  ${MyImageName} \
  --input /data/input/test.txt \
  --output /data/output/pred.csv
```

基本镜像为`pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel`，构建时通过`pip install transformers && pip install pandas && pip install scikit-learn`安装依赖包即可运行。容器的WORKDIR设定为`/app/sohu_matching/src`（由于代码采取相对路径，不在此目录运行会报错；如果测试命令运行出错，可进入该目录直接`python infer_final.py`并指定输入与输出文件位置）。镜像大小约10G，其中`sohu_matching/checkpoints/rematch`存放我们的模型，总大小在2G以内，符合比赛要求。

#### 简介

本项目包含了我们在2021搜狐校园文本匹配**复赛环节**的PyTorch版本代码，在复赛Public排行榜上排名第三，线上测评的F1分数为0.791075301658579，其中A类任务0.8548399419359769，B类任务0.727310661381181。

我们采用了联合训练的方式，在A、B两个任务上采用一个共同的基于预训练语言模型的encoder，而后分别为各个任务采用多组简单的全连接结构作为classifier。我们使用了不同的预训练模型（如NEZHA、MacBert、ROBERTA、ERNIE等），设计了选择了两种文本匹配的技术路线（通过[SEP]拼接source与target作为输入、类似SBERT的句子向量编码进行比较），并尝试了多种上分策略（如在给定语料上继续mlm预训练、focal loss损失函数、不同的pooling策略、加入TextCNN、fgm对抗训练、数据增强等）。我们选取了多组差异较大的模型的输出，通过投票的方式进行集成，得到最好成绩。

#### 项目结构

```bash
│  README.md				# README
│  test.yaml				# conda环境配置
│  							# 基本上安装pytorch>=1.6和transformer即可复现
├─checkpoints				# 用于保存模型
├─data						
│  └─dummy_bert				# 包含BERT\ERNIE\NEZHA的分词词表及config.json
│  └─dummy_ernie			# 用于模型推断时从config文件定义模型，不加载原预训练权重
│  └─dummy_nezha
│  └─sohu2021_open_data		# 包含初赛及复赛的训练、评估和测试数据
│      ├─短短匹配A类			 # 包括train.txt, train_r2.txt, train_r3.txt, train_rematch.txt
│      ├─短短匹配B类			 # valid.txt, valid_rematch.txt, test_with_id_rematch.txt
│      ├─短长匹配A类
│      ├─短长匹配B类
│      ├─长长匹配A类
│      └─长长匹配B类
├─logs							# 用于保存日志，例：python train.py > log_dir
├─results						# 用于保存测试集推理结果
├─valid_output					# 记录模型在valid上的输出，并计算各类f1  
└─src							# 主要代码文件夹
    │  config.py				# 模型与训练等参数统一通过config.py设置
    │  data.py					# 数据读取，DataLoader等
    │  infer.py					# 测试集推理代码
    │  merge_result.py			# 用于投票集成
    │  model.py					# 模型定义
    │  search_better_merge.py	# 在验证集输出上寻找最优投票组合
    │  train.py					# 训练代码，支持多任务形式（更改model中的num_task）
    │  train_old.py				# 训练代码，仅支持A\B两任务，复赛中主要使用该方式训练模型
    │  utils.py					# 其他函数等
    │  
    ├─new_runs					# tensorboard事件目录，用于可视化损失函数等指标
    ├─NEZHA						# nezha相关的模型结构定义等
    │  │  model_nezha.py	
    │  │  nezha_utils.py     
    └─__pycache__
```

#### 运行示例
(备注：决赛提交中针对A\B类测试样本在同一个文件中的情况略微修改了`data.py`，直接运行`train_old.py`可能会有错误)
补充训练数据后，在`config.py`文件中设置训练相关参数，进入到src文件夹下，运行`train_old.py`进行训练（在复赛中，我们尝试了为6个子任务分别设置分类网络的形式，统一在`train.py`中，但对于A\B两任务的情况，初赛训练代码方式效果似乎更加，因此我们在`train_old.py`中保留了原方式，并作为主要训练代码；默认多卡训练，在`train_old.py`调整设备卡数），可通过重定向将输出保存为日志。训练结束后，在`config.py`中设置推理相关参数，进入到src文件夹下，运行`infer.py`进行推理（默认多卡推理，在`infer.py`调整设备卡数）。

```bash
python train_old.py > ../logs/0523/0523_roberta_80k.log	# 训练并保存输出日志
python infer.py		# 推理
```

