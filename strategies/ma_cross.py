"""
双均线交叉策略 | Dual Moving Average Crossover
经典入门策略：短期均线上穿长期均线 → 买入，下穿 → 卖出
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class MACrossover:
    """双均线交叉策略"""
    
    def __init__(self, short_window: int = 5, long_window: int = 20):
        self.short_window = short_window
        self.long_window = long_window
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成买卖信号"""
        df = df.copy()
        df['ma_short'] = df['close'].rolling(window=self.short_window).mean()
        df['ma_long'] = df['close'].rolling(window=self.long_window).mean()
        df['signal'] = 0
        df.loc[df['ma_short'] > df['ma_long'], 'signal'] = 1
        df['position'] = df['signal'].diff()
        return df
    
    def backtest(self, df: pd.DataFrame, capital: float = 100000) -> dict:
        """简单回测"""
        df = self.generate_signals(df)
        df['returns'] = df['close'].pct_change()
        df['strategy_returns'] = df['signal'].shift(1) * df['returns']
        df['cumulative'] = (1 + df['strategy_returns']).cumprod() * capital
        
        total_return = (df['cumulative'].iloc[-1] / capital - 1) * 100
        sharpe = df['strategy_returns'].mean() / df['strategy_returns'].std() * np.sqrt(252)
        max_drawdown = (df['cumulative'] / df['cumulative'].cummax() - 1).min() * 100
        win_rate = (df['strategy_returns'] > 0).mean() * 100
        
        return {
            'total_return_pct': round(total_return, 2),
            'sharpe_ratio': round(sharpe, 2),
            'max_drawdown_pct': round(max_drawdown, 2),
            'win_rate_pct': round(win_rate, 2)
        }
    
    def plot(self, df: pd.DataFrame, save_path: str = None):
        """可视化"""
        df = self.generate_signals(df)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        ax1.plot(df.index, df['close'], label='Close', alpha=0.7, linewidth=1)
        ax1.plot(df.index, df['ma_short'], label=f'MA{self.short_window}', alpha=0.8)
        ax1.plot(df.index, df['ma_long'], label=f'MA{self.long_window}', alpha=0.8)
        
        # 标注买卖点
        buy_signals = df[df['position'] == 2]
        sell_signals = df[df['position'] == -2]
        ax1.scatter(buy_signals.index, buy_signals['close'], marker='^', 
                    color='green', s=100, label='Buy', alpha=0.8)
        ax1.scatter(sell_signals.index, sell_signals['close'], marker='v', 
                    color='red', s=100, label='Sell', alpha=0.8)
        
        ax1.legend()
        ax1.set_title(f'MA Crossover ({self.short_window}/{self.long_window})')
        
        df['cumulative'] = (1 + df['strategy_returns']).cumprod()
        ax2.plot(df.index, df['cumulative'], label='Strategy', color='green')
        ax2.set_title('Cumulative Returns')
        
        if save_path:
            plt.savefig(save_path)
        plt.show()


if __name__ == "__main__":
    from data.fetch_data import fetch_a_stock
    
    # 获取平安银行数据
    df = fetch_a_stock("000001", start="20220101", end="20251231")
    print(f"数据量: {len(df)}")
    
    strategy = MACrossover(short_window=5, long_window=20)
    results = strategy.backtest(df)
    
    print("\n======= 回测结果 =======")
    for k, v in results.items():
        print(f"  {k}: {v}")
