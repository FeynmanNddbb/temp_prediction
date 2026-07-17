<h1 align="center">🌡️ 基于空气污染感知的智能温度预测系统</h1>

<p align="center">
  <em>Air-Pollution-Aware Intelligent Temperature Prediction System</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.6+-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" />
</p>

---

## 📖 项目简介

本系统采用 **双层 LSTM 深度学习网络**，通过分析历史空气质量传感器数据（CO、苯、NO₂、NOₓ、相对湿度、绝对湿度等），实现对环境温度的实时智能预测。配备交互式可视化界面，支持手动输入与批量文件导入两种预测模式。

> 数据来源：[UCI Air Quality Dataset](https://archive.ics.uci.edu/ml/datasets/Air+Quality)

---

## ✨ 功能亮点

| 功能 | 说明 |
|------|------|
| 🧠 **LSTM 预测** | 双层 LSTM + Dropout，捕捉长期时序依赖 |
| 🖱️ **手动预测** | 实时输入6项环境参数，即时获得预测结果 |
| 📂 **文件导入** | 支持 `.xlsx` / `.xls` 批量数据导入预测 |
| 📈 **交互图表** | matplotlib 绘制预测 vs 实际温度对比曲线 |
| 💾 **结果导出** | 导出预测历史为 CSV 或 Excel 文件 |
| 🔄 **数据平滑** | 5 点均匀滑动窗口，曲线更平滑美观 |

---

## 🗂️ 项目结构

```
temp_prediction_last/
├── train.py              # 模型训练入口
├── predict_input.py      # 预测 GUI 应用入口
├── requirements.txt      # Python 依赖清单
├── 用户手册.md            # 详细用户操作手册
├── 软著/                 # 软件著作权相关材料
│
├── AirQualityUCI.csv     # 训练数据集（需自行准备，见下文）
│
└── [训练后自动生成]
    ├── model_weights.pt  # 最佳模型权重
    ├── scaler_X.pkl      # 特征标准化器
    ├── scaler_y.pkl      # 标签标准化器
    ├── X_test.pt         # 测试集特征张量
    └── y_test.pt         # 测试集标签张量
```

---

## 🚀 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/<your-username>/temp-prediction.git
cd temp-prediction
```

### 2. 创建虚拟环境（推荐）

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 准备数据集

下载 [AirQualityUCI.csv](https://archive.ics.uci.edu/ml/datasets/Air+Quality) 并放置于项目根目录。  
数据集应包含以下列（以分号分隔，逗号为小数点）：

```
CO(GT);C6H6(GT);NO2(GT);NOx(GT);RH;AH;T
```

### 5. 训练模型

```bash
python train.py
```

训练 100 个 epoch，完成后自动保存 `model_weights.pt`、`scaler_X.pkl`、`scaler_y.pkl`、`X_test.pt`、`y_test.pt`。

### 6. 启动预测界面

```bash
python predict_input.py
```

---

## 🏗️ 模型架构

```
输入序列 (24 × 7)
    │
    ▼
┌─────────────────────────┐
│  LSTM Layer 1            │  hidden_size=128, dropout=0.2
│  LSTM Layer 2            │
└─────────────────────────┘
    │  取最后一个时间步
    ▼
┌─────────────────────────┐
│  Fully Connected (128→1) │
└─────────────────────────┘
    │
    ▼
预测温度 (°C)
```

| 超参数 | 值 |
|--------|-----|
| 输入特征 | 7（CO, C6H6, NO₂, NOₓ, RH, AH, T） |
| 时间窗口 | 24 个时间步 |
| 隐藏层大小 | 128 |
| LSTM 层数 | 2 |
| Dropout | 0.2 |
| 优化器 | Adam (lr=0.001) |
| 学习率调度 | ReduceLROnPlateau (factor=0.5, patience=5) |
| 损失函数 | MSE |
| 训练轮数 | 100 epochs |
| 数据划分 | 训练 70% / 验证 15% / 测试 15% |

---

## 🖥️ 界面预览

```
┌─────────────────────────────────────────────────────┐
│          基于空气污染感知的智能温度预测系统            │
│  ┌───────────────────────────────────────────────┐  │
│  │          温度预测结果对比  (交互图表)           │  │
│  │   ~~预测温度(蓝虚线)~~  —实际温度(红实线)—     │  │
│  └───────────────────────────────────────────────┘  │
│  当前环境参数输入                                      │
│  [ CO ]  [C6H6]  [NO2]  [NOx]  [ RH ]  [ AH ]       │
│  [手动预测]  [文件导入]  [保存结果]                    │
│              预测温度：20.35 °C                       │
└─────────────────────────────────────────────────────┘
```

---

## 📦 依赖环境

| 包 | 版本要求 |
|----|---------|
| Python | ≥ 3.6 |
| PyTorch | ≥ 2.0 |
| scikit-learn | ≥ 1.3 |
| pandas | ≥ 2.0 |
| numpy | ≥ 1.24 |
| matplotlib | ≥ 3.7 |
| scipy | ≥ 1.11 |
| joblib | ≥ 1.3 |
| openpyxl | ≥ 3.1 |

---

## ⚙️ Tkinter 配置说明

大多数 Python 安装均自带 Tkinter，无需额外配置。  
若启动时报 `_tkinter` 相关错误，请在 `predict_input.py` 顶部手动指定路径：

```python
import os
os.environ["TCL_LIBRARY"] = r"<Python安装路径>\tcl\tcl8.6"
os.environ["TK_LIBRARY"]  = r"<Python安装路径>\tcl\tk8.6"
```

---

## 📄 数据格式

### 训练数据（CSV）

```csv
CO(GT);C6H6(GT);NO2(GT);NOx(GT);RH;AH;T
1.50;5.00;80.00;120.00;45.00;0.50;18.50
```

### 预测导入文件（Excel）

| CO | C6H6 | NO2 | NOx | RH | AH |
|----|------|-----|-----|----|----|
| 1.5 | 5.0 | 80 | 120 | 45 | 0.5 |

- 列名需与上表完全一致（区分大小写）
- 至少 10 行有效数值数据，无缺失值

---

## 📜 许可证

本项目仅供学习与研究使用，版权所有 © 2024。

---

<p align="center">Made with ❤️ using PyTorch + Tkinter</p>
