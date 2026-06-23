# CarSpec AI — 车辆多属性智能识别系统

> Module 2 Project · Computer Vision · 基于 CompCars 数据集

## 项目简介

CarSpec AI 是一个基于计算机视觉的车辆多属性识别系统。上传一张车辆外观照片，系统同时预测 **车型类型**、**门数**、**座位数** 三个属性，并提供可解释的视觉特征分析。

## 核心创新

1. **多任务联合学习**：ResNet50 共享 backbone + 三个分类头，利用任务间相关性提升性能
2. **可解释视觉特征**：提取 50+ 手工特征（颜色直方图、HOG、纹理、车身比例、对称性），提供比纯 CNN 更好的可解释性
3. **三模型对比**：Naive 基线 / Classical ML（随机森林）/ Deep Learning（ResNet50 多任务）

## 项目结构

```
├── README.md
├── requirements.txt
├── Makefile
├── setup.py                <- 训练管线
├── main.py                 <- Flask Web 应用
├── Dockerfile
├── scripts/
│   ├── data.py             <- 数据加载
│   ├── make_dataset.py     <- 数据下载
│   ├── features.py         <- 可解释特征提取
│   ├── model.py            <- 三个模型实现
│   └── experiment.py       <- 实验框架
├── models/                 <- 训练好的模型
├── data/
│   ├── raw/                <- 原始数据
│   ├── processed/          <- 处理后数据
│   └── outputs/            <- 输出结果
├── static/                 <- 前端资源
├── templates/              <- HTML 模板
├── notebooks/              <- 探索性分析
└── .github/                <- CI/PR 模板
```

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 下载数据
```bash
python -m scripts.make_dataset
```
按提示从 http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/ 下载数据，解压到 `data/raw/compcars/`。

### 3. 训练模型
```bash
python setup.py
```

### 4. 启动应用
```bash
python main.py
```
访问 http://localhost:7860

## 三个模型位置

| 模型 | 代码位置 | 训练后模型 |
|------|---------|-----------|
| Naive 基线 | `scripts/model.py` → `NaiveBaseline` 类 | `models/naive_*.pkl` |
| Classical ML | `scripts/model.py` → `ClassicalModel` 类 | `models/classical_*.pkl` |
| Deep Learning | `scripts/model.py` → `DeepMultiTaskModel` 类 | `models/deep_multitask.h5` |

## 数据集

**CompCars** (CVPR 2015)：136,726 张整车图片，标注车型类型、门数、座位数、最大速度、排量等属性。

- 官网: http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/
- 论文: Yang et al., "A Large-Scale Car Dataset for Fine-Grained Categorization and Verification"

## 部署

应用部署在 HuggingFace Spaces（Docker SDK）：
- 在线 Demo: [部署后填写 URL]

## 技术栈

- **深度学习**: TensorFlow/Keras, ResNet50
- **经典 ML**: scikit-learn, 随机森林
- **特征工程**: scikit-image (HOG), OpenCV-style 特征
- **Web 框架**: Flask
- **部署**: Docker, HuggingFace Spaces
