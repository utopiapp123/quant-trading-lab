# 🧪 Quant Trading Lab | 量化交易练手实验室

经典量化策略实现、回测框架、因子分析、机器学习预测。从零开始学量化。

## 📂 项目结构

```
quant-trading-lab/
├── strategies/              # 经典策略实现
│   ├── ma_cross.py          # 双均线交叉策略
│   ├── bollinger.py         # 布林带突破策略
│   └── momentum.py          # 动量轮动策略
├── backtest/                # 回测引擎
│   └── simple_backtest.py   # 事件驱动回测框架
├── factors/                 # 因子分析
│   └── factor_analysis.py   # 多因子选股（动量/估值/质量/成交量）
├── ml/                      # 机器学习
│   └── lstm_predictor.py    # LSTM 股价涨跌预测
├── data/                    # 数据获取
│   └── fetch_data.py        # A股/指数数据拉取 (AKShare)
├── requirements.txt
└── README.md
```

## 🚀 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 获取数据
python data/fetch_data.py

# 跑个策略
python strategies/ma_cross.py

# 跑个回测
python backtest/simple_backtest.py

# 跑个因子分析
python factors/factor_analysis.py

# 训练 LSTM 预测模型
python ml/lstm_predictor.py
```

## 📋 功能清单

| 模块 | 文件 | 功能 | 难度 |
|------|------|------|------|
| 双均线交叉 | strategies/ma_cross.py | 5/20日线金叉死叉 + 回测 + 可视化 | ⭐ |
| 布林带突破 | strategies/bollinger.py | 2σ布林带超买超卖信号 | ⭐⭐ |
| 动量轮动 | strategies/momentum.py | 多标的动量排名 + 轮动选股 | ⭐⭐ |
| 回测引擎 | backtest/simple_backtest.py | 事件驱动、滑点、手续费、仓位管理 | ⭐⭐⭐ |
| 因子分析 | factors/factor_analysis.py | 动量/估值/质量/成交量四维因子 | ⭐⭐⭐ |
| LSTM预测 | ml/lstm_predictor.py | LSTM + 多特征工程 + 涨跌分类 | ⭐⭐⭐⭐ |

## 📊 回测引擎特性

- 事件驱动架构
- 滑点 & 手续费模拟
- 多标的组合管理
- 权益曲线追踪
- 自动报告生成 (年化收益、Sharpe、最大回撤、胜率)

## ⚠️ 免责声明

本项目仅供学习研究，不构成投资建议。量化交易有风险，实盘需谨慎。
