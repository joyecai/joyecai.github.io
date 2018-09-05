---
layout:     post
title:      Attention Model
subtitle:   注意力机制
date:       2018-09-05
author:     Jiayue Cai
header-img: img/post-bg-debug.png
catalog: true
tags:
    - Machine Learning
---


>Last updated on 2018-9-5... 

### 介绍

Google 2017年论文[Attention is All you need](https://arxiv.org/pdf/1706.03762.pdf)中，为Attention做了如下定义：
![Attention](https://upload-images.jianshu.io/upload_images/13187322-0904d285d5835ed6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/315/format/webp)
其中Q代表`query`、K代表`key`、V代表`value`，d为缩放因子

计算Attention Weighted Value的流程图：
![compute](https://upload-images.jianshu.io/upload_images/13187322-bd743638ad420f2c.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/568/format/webp)
计算Q、K相似度 `->` 得分归一化(Attention Weight) `->` 根据得分对V进行加权

### 发展

Attention机制最早是在视觉图像领域提出来的(上世纪90年代)，但是真正热门起来是由google mind团队于2014年的论文《Recurrent Models of Visual Attention》，他们在RNN模型上使用了Attention机制来进行图像分类。

随后，Bahdanau等人在论文《Neural Machine Translation by Jointly Learning to Align and Translate》中，使用类似attention的机制在机器翻译任务上将翻译和对齐同时进行，他们的工作算是第一个将Attention机制应用到NLP领域中。

接着Attention机制被广泛应用在基于RNN/CNN等神经网络模型的各种NLP任务中。2017年，google机器翻译团队发表的《Attention is all you need》中大量使用了自注意力（self-attention）机制来学习文本表示。

自注意力机制也成为了大家近期的研究热点，并在各种NLP任务上进行探索。下图展示了Attention研究进展的大概趋势：
![dev](https://upload-images.jianshu.io/upload_images/13187322-2b66324483d782fe.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/568/format/webp)

### 应用

#### 学习权重分布

- 这个加权可以是保留`所有分量`均做加权（即soft attention）；也可以是在分布中以某种采样策略选取`部分分量`（即hard attention），此时常用RL来做
- 这个加权可以作用在`原图`上，也就是《Recurrent Model of Visual Attention》（RAM）和《Multiple Object Recognition with Visual Attention》（DRAM）；也可以作用在`特征图`上，如后续的好多文章（例如image caption中的《 Show, Attend and Tell: Neural Image Caption Generation with Visual Attention》) 
- 这个加权可以作用在`空间尺度`上，给不同空间区域加权；也可以作用在`channel尺度`上，给不同通道特征加权；甚至`特征图`上每个元素加权
- 这个加权还可以作用在`不同时刻历史特征`上，如Machine Translation

#### 任务聚焦/解耦（通过attention mask）

通过将任务分解，设计不同的网络结构（或分支）专注于不同的子任务，重新分配网络的学习能力，从而降低原始任务的难度，使网络更加容易训练。

- 多任务模型：通过Attention对feature进行权重再分配，聚焦各自关键特征
- 多步负荷预测：（多任务多输出模型）每步预测对于特征的关注点应该不一样，学习一个feature mapping的mask attention
- 异常数据mask负荷预测：在原始feature mapping 后接一个attention，自动mask异常输入，提升模型的鲁棒性

### 分类

1、按输出
- soft attention：保留所有分量均做加权，输出注意力分布的概率值
- hard attention：在分布中以某种采样策略选取部分分量，输出onehot向量（强化学习）

2、按关注的范围
- Globle attention：全局注意力顾名思义对整个feature mapping进行注意力加权
- Local attention：局部注意力有两种，第一种首先通过一个hard-globle-attention锁定位置，在位置上下某个local窗口进行注意力加权

3、按score函数
- 点		积：Similarity(Query,Key<sub>i</sub>) = Query*Key<sub>i</sub>
- Cosine相似性：Similarity(Query,Key<sub>i</sub>) = \frac{Query*Key<sub>i</sub>}{\Vert{Query}*\Vert{Key<sub>i</sub>}} 
- M L P 网	络：Similarity(Query,Key<sub>i</sub>) = MLP(Query,Key<sub>i</sub>)




