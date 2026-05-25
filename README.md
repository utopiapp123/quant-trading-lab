# 🧪 Quant Trading Lab | 量化交易练手实验室

经典量化策略实现、回测框架、因子分析。从零开始学量化。

## 📂 项目结构

```
quant-trading-lab/
├── strategies/          # 经典策略实现
│   ├── ma_cross.py      # 双均线交叉策略
│   ├── bollinger.py     # 布林带突破策略
│   └── momentum.py      # 动量轮动策略
├── backtest/            # 回测引擎
│   └── simple_backtest.py
├── factors/             # 因子分析
│   └── factor_analysis.py
├── ml/                  # 机器学习
│   └── lstm_predictor.py  # LSTM 股价预测
├── data/                # 数据获取脚本
│   └── fetch_data.py
└── requirements.txt
```

## 🚀 快速开始

```bash
pip install -r requirements.txt
export AKSHARE_DATA_DIR=./data
python strategies/ma_cross.py
```

## 📋 策略清单

| 策略 | 文件 | 回测收益 | 难度 |
|------|------|----------|------|
| 双均线交叉 | ma_cross.py | ~15% 年化 | ⭐ |
| 布林带突破 | bollinger.py | ~12% 年化 | ⭐⭐ |
| 动量轮动 | momentum.py | ~18% 年化 | ⭐⭐ |
| LSTM 预测 | lstm_predictor.py | - | ⭐⭐⭐⭐ |

## ⚠️ 免责声明

本项目仅供学习研究，不构成投资建议。量化交易有风险，实盘需谨慎。
