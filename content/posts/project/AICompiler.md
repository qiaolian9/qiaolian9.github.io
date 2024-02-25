---
title: "AICompiler"
date: 2023-12-04T23:52:23+08:00 #创建时间
lastmod: 2024-02-23T23:52:23+08:00 #更新时间
author: ["Brocoli"]
keywords: 
- 
categories: # 没有分类界面可以不填写
- 
tags: # 标签
- AICompiler
description: ""
weight: 1
slug: ""
draft: false # 是否为草稿
comments: true # 本页面是否显示评论
reward: false # 打赏
mermaid: true #是否开启mermaid
showToc: true # 显示目录
TocOpen: true # 自动展开目录
hidemeta: false # 是否隐藏文章的元信息，如发布日期、作者等
disableShare: true # 底部不显示分享栏
showbreadcrumbs: true #顶部显示路径
cover:
    image: "" #图片路径例如：posts/tech/123/123.png
    zoom: # 图片大小，例如填写 50% 表示原图像的一半大小
    caption: "" #图片底部描述
    alt: ""
    relative: false

---

## Torch2Tensor

_2023.3~2023.5_

**Torch nn.Module** -> **TVM Tensor program**

> Github: <https://github.com/qiaolian9/Torch2Tensor>

* A easy tool for generating Tensor Program from Torch(besd on Torch FX & TVM Relax)

## Pruner

_2023.5~2024.1_

**Key:** Deep Learning Compiler, Tensor Optimization

Pruner: An Efficient Cross-Platform Tensor Compiler with Dual Awareness

> URL:<https://arxiv.org/abs/2402.02361>
>
> Github: <https://github.com/qiaolian9/Pruner>

* Tensor program optimization on Deep Learning Accelerators (DLAs) is critical for efficient model deployment. Although search-based Deep Learning Compilers (DLCs) have achieved significant performance gains compared to manual methods, they still suffer from the persistent challenges of low search efficiency and poor cross-platform adaptability. In this paper, we propose **Pruner**, following hardware/software co-design principles to hierarchically boost tensor program optimization. Pruner comprises two primary components: a Parameterized Static Analyzer (**PSA)** and a Pattern-aware Cost Model (**PaCM**). The former serves as a hardware-aware and formulaic performance analysis tool, guiding the pruning of the search space, while the latter enables the performance prediction of tensor programs according to the critical data-flow patterns. Furthermore, to ensure effective cross-platform adaptation, we design a Momentum Transfer Learning (**MTL**) strategy using a Siamese network, which establishes a bidirectional feedback mechanism to improve the robustness of the pre-trained cost model. The extensive experimental results demonstrate the effectiveness and advancement of the proposed Pruner in various tensor program tuning tasks across both online and offline scenarios, with low resource overhead.


