"""
动量轮动策略 | Momentum Rotation
买入近期最强标的，定期调仓
"""
import pandas as pd
import numpy as np

class MomentumRotation:
    def __init__(self, lookback: int = 20, top_n: int = 3):
        self.lookback = lookback
        self.top_n = top_n
    
    def calc_momentum(self, df: pd.DataFrame) -> float:
        return df['close'].pct_change(self.lookback).iloc[-1]
    
    def rank(self, dfs: dict) -> pd.DataFrame:
        scores = {name: self.calc_momentum(df) for name, df in dfs.items()}
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        print("动量排名:")
        for i, (name, score) in enumerate(ranked, 1):
            flag = "✅" if i <= self.top_n else ""
            print(f"  {i}. {name}: {score:.2%} {flag}")
        selected = [name for name, _ in ranked[:self.top_n]]
        return pd.DataFrame({'selected': selected})

if __name__ == "__main__":
    from data.fetch_data import fetch_a_stock
    stocks = {'平安银行':'000001','万科A':'000002','五粮液':'000858','茅台':'600519','平安':'601318'}
    dfs = {name: fetch_a_stock(code) for name, code in stocks.items()}
    m = MomentumRotation(lookback=20, top_n=2)
    print(f"\n入选: {list(m.rank(dfs)['selected'])}")
