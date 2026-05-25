"""
布林带策略 | Bollinger Bands
价格触及下轨 → 超卖买入，触及上轨 → 超买卖出
"""
import pandas as pd
import numpy as np

class BollingerStrategy:
    def __init__(self, window: int = 20, num_std: float = 2.0):
        self.window = window
        self.num_std = num_std
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['ma'] = df['close'].rolling(self.window).mean()
        df['std'] = df['close'].rolling(self.window).std()
        df['upper'] = df['ma'] + self.num_std * df['std']
        df['lower'] = df['ma'] - self.num_std * df['std']
        df['bandwidth'] = (df['upper'] - df['lower']) / df['ma'] * 100
        df['signal'] = 0
        df.loc[df['close'] < df['lower'], 'signal'] = 1
        df.loc[df['close'] > df['upper'], 'signal'] = -1
        df['position'] = df['signal'].diff()
        return df
    
    def backtest(self, df: pd.DataFrame, capital: float = 100000) -> dict:
        df = self.generate_signals(df)
        df['returns'] = df['close'].pct_change()
        df['strategy_returns'] = df['signal'].shift(1) * df['returns']
        df['cumulative'] = (1 + df['strategy_returns']).cumprod() * capital
        return {
            'total_return_pct': round((df['cumulative'].iloc[-1]/capital - 1)*100, 2),
            'sharpe_ratio': round(df['strategy_returns'].mean()/df['strategy_returns'].std()*np.sqrt(252), 2),
            'max_drawdown_pct': round((df['cumulative']/df['cumulative'].cummax()-1).min()*100, 2),
        }

if __name__ == "__main__":
    from data.fetch_data import fetch_a_stock
    df = fetch_a_stock("000001")
    s = BollingerStrategy()
    r = s.backtest(df)
    for k, v in r.items():
        print(f"  {k}: {v}")
