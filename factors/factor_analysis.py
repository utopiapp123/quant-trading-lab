"""
因子分析 | Factor Analysis Suite
多因子选股：估值、动量、质量、波动率四大类因子
"""
import pandas as pd
import numpy as np

class FactorAnalyzer:
    """多因子分析器"""
    
    def __init__(self):
        self.factors = {}
    
    def calc_momentum_factors(self, df: pd.DataFrame) -> dict:
        """动量类因子"""
        close = df['close']
        return {
            'momentum_5d': close.pct_change(5).iloc[-1],
            'momentum_10d': close.pct_change(10).iloc[-1],
            'momentum_20d': close.pct_change(20).iloc[-1],
            'momentum_60d': close.pct_change(60).iloc[-1],
            'rsi': self._calc_rsi(close, 14).iloc[-1],
        }
    
    def calc_value_factors(self, df: pd.DataFrame) -> dict:
        """估值类因子 (基于价格特征替代)"""
        close, high, low = df['close'], df['high'], df['low']
        ma20 = close.rolling(20).mean()
        return {
            'price_to_ma20': close.iloc[-1] / ma20.iloc[-1],
            'price_to_ma60': close.iloc[-1] / close.rolling(60).mean().iloc[-1],
            'max_drawdown_60d': (close.rolling(60).max().iloc[-1] / close.iloc[-1] - 1),
            'high_low_spread': (high.iloc[-1] - low.iloc[-1]) / close.iloc[-1],
        }
    
    def calc_quality_factors(self, df: pd.DataFrame) -> dict:
        """质量类因子"""
        close, volume = df['close'], df['volume']
        returns = close.pct_change()
        return {
            'sharpe_60d': returns.rolling(60).mean() / returns.rolling(60).std() * np.sqrt(252),
            'volatility_20d': returns.rolling(20).std(),
            'avg_volume_ratio': volume.iloc[-1] / volume.rolling(20).mean().iloc[-1],
            'up_days_ratio': (returns > 0).rolling(60).mean().iloc[-1],
            'return_stability': 1 - returns.rolling(60).std() / abs(returns.rolling(60).mean() + 1e-9),
        }
    
    def calc_volume_factors(self, df: pd.DataFrame) -> dict:
        """成交量因子"""
        volume = df['volume']
        vma5 = volume.rolling(5).mean()
        vma20 = volume.rolling(20).mean()
        return {
            'volume_ratio_5': volume.iloc[-1] / vma5.iloc[-1],
            'volume_ratio_20': volume.iloc[-1] / vma20.iloc[-1],
            'volume_trend': (vma5.iloc[-1] / vma20.iloc[-1] - 1),
            'volume_std_20': volume.rolling(20).std().iloc[-1] / vma20.iloc[-1],
        }
    
    def analyze_single(self, df: pd.DataFrame, name: str = "Unknown") -> pd.DataFrame:
        """单标的因子分析"""
        all_factors = {}
        all_factors.update(self.calc_momentum_factors(df))
        all_factors.update(self.calc_value_factors(df))
        all_factors.update(self.calc_quality_factors(df))
        all_factors.update(self.calc_volume_factors(df))
        
        result = pd.DataFrame([all_factors], index=[name]).T
        result.columns = ['value']
        
        # 因子方向标注
        direction_map = {
            'momentum_20d': 1, 'rsi': 1, 'sharpe_60d': 1,
            'up_days_ratio': 1, 'price_to_ma20': -1, 'max_drawdown_60d': -1,
            'volatility_20d': -1,
        }
        result['direction'] = result.index.map(lambda x: direction_map.get(x, 1))
        result['score'] = result['value'] * result['direction']
        
        return result
    
    def rank_stocks(self, dfs: dict) -> pd.DataFrame:
        """多标的因子排名"""
        scores = {}
        for name, df in dfs.items():
            analysis = self.analyze_single(df, name)
            scores[name] = analysis['score'].mean()
        
        ranked = pd.DataFrame({'score': scores}).sort_values('score', ascending=False)
        ranked['rank'] = range(1, len(ranked) + 1)
        return ranked
    
    def _calc_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

if __name__ == "__main__":
    from data.fetch_data import fetch_a_stock
    
    # 多标的分析
    stocks = {'平安银行':'000001','万科A':'000002','五粮液':'000858',
              '茅台':'600519','平安':'601318','招行':'600036'}
    
    dfs = {}
    for name, code in stocks.items():
        try:
            dfs[name] = fetch_a_stock(code)
        except:
            pass
    
    analyzer = FactorAnalyzer()
    
    # 单标的详细因子
    print("======= 平安银行 因子详情 =======")
    result = analyzer.analyze_single(dfs['平安银行'], '平安银行')
    print(result[['value']].round(4))
    
    # 多标的排名
    print("\n======= 因子综合排名 =======")
    ranked = analyzer.rank_stocks(dfs)
    print(ranked.round(4))
