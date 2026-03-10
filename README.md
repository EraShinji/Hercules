# HERCULES: HEart Rhythm Classification via Unsupervised LEarning for Smartwatches

**H**Eart **R**hythm **C**lassification via **U**nsupervised **L**earning for Smartwatch**ES**

基于对比学习的ECG心电信号分类模型，支持在Apple Watch等边缘设备上部署。

## 项目概述

本项目采用两阶段训练策略，旨在从大规模I导联心电数据中学习有效的特征表示，用于潜在心脏疾病的自动鉴别。

### 核心特性

- **对比学习预训练**：无监督学习心电信号的有效Embedding表示
- **下游任务微调**：基于预训练权重进行疾病分类
- **边缘设备部署**：模型轻量化设计，支持Apple Watch等可穿戴设备
- **多格式支持**：支持MAT、DAT等常见心电数据格式

## 技术架构

### 阶段一：对比学习预训练 (Self-Supervised Learning)

```
输入: I导联ECG信号 (10万条无标注数据)
    ↓
数据增强 → 生成正样本对 (同一样本的不同增强视图)
         → 生成负样本对 (不同样本)
    ↓
Encoder网络 → 学习Embedding表示
    ↓
对比损失 (InfoNCE) → 最小化正样本距离，最大化负样本距离
    ↓
输出: 预训练Encoder权重
```

**目标**：学习对下游任务有益的心电信号特征表示，无需人工标注。

### 阶段二：下游分类微调 (Fine-tuning)

```
输入: 带标注的ECG数据
    ↓
加载预训练Encoder (冻结/微调)
    ↓
添加分类头 (Classification Head)
    ↓
监督训练 → 疾病分类任务
    ↓
输出: 疾病预测模型
```

**目标**：利用预训练学到的特征，在小样本标注数据上实现高精度分类。

## 项目结构

```
.
├── module/
│   └── utils/
│       └── format_convertion/    # 数据格式转换工具
│           ├── convertion.py     # 抽象基类
│           ├── mat2tensor.py     # MAT格式转Tensor
│           └── dat2tensor.py     # DAT格式转Tensor
├── data/
│   └── for_test/
│       └── mat/                  # 测试数据
├── pyproject.toml                # 项目依赖
└── README.md                     # 项目说明
```

## 数据格式支持

| 格式 | 说明 | 状态 |
|------|------|------|
| MAT | MATLAB格式心电数据 | ✅ 支持 |
| DAT | WFDB格式心电数据 | ✅ 支持 |
| HEA | 头文件（含导联信息、患者元数据） | ✅ 支持 |

### 数据加载示例

```python
from module.utils.format_convertion.mat2tensor import Mat2Tensor

# 加载I导联数据
converter = Mat2Tensor("path/to/record", lead_name="I")

# 访问心电信号Tensor
signal_tensor = converter.dat_tensor  # shape: [N]

# 访问患者元信息
print(converter.hea_comments)
# {'Age': '85', 'Sex': 'Male', 'Dx': '164889003,59118001,...', ...}
```

## 环境依赖

- Python 3.14
- PyTorch
- scipy
- wfdb

安装依赖：

```bash
uv sync
```

## 训练规模

- **数据量**：10万条I导联心电记录
- **目标平台**：Apple Watch (边缘设备部署)
- **应用场景**：实时心电监测与潜在疾病预警

## 路线图

- [x] 数据格式转换工具 (MAT/DAT → Tensor)
- [x] HEA文件解析与单导联提取
- [ ] 对比学习预训练框架
- [ ] 下游分类微调模块
- [ ] 模型量化与边缘部署优化
- [ ] Apple Watch集成Demo

## 许可证

MIT-License