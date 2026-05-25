"""
回测引擎 | Backtest Engine
事件驱动的简易回测框架，支持组合管理、滑点、手续费
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Callable

@dataclass
class Order:
    symbol: str
    action: str  # 'buy' or 'sell'
    amount: float
    price: float
    date: pd.Timestamp

@dataclass  
class Trade:
    symbol: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    shares: int
    pnl: float
    pnl_pct: float

class BacktestEngine:
    """简易回测引擎"""
    
    def __init__(self, initial_capital: float = 100000, commission: float = 0.0003, 
                 slippage: float = 0.001):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.cash = initial_capital
        self.positions: Dict[str, int] = {}
        self.orders: List[Order] = []
        self.trades: List[Trade] = []
        self.equity_curve = []
        self._open_trades: Dict[str, Trade] = {}
    
    def reset(self):
        self.cash = self.initial_capital
        self.positions = {}
        self.orders = []
        self.trades = []
        self.equity_curve = []
        self._open_trades = {}
    
    def order(self, symbol: str, action: str, amount: float, price: float, 
              date: pd.Timestamp):
        """下单（含滑点）"""
        slip = price * self.slippage
        exec_price = price + slip if action == 'buy' else price - slip
        self.orders.append(Order(symbol, action, amount, exec_price, date))
        
        if action == 'buy':
            shares = int(amount / exec_price)
            cost = shares * exec_price * (1 + self.commission)
            if cost <= self.cash:
                self.cash -= cost
                self.positions[symbol] = self.positions.get(symbol, 0) + shares
                # 开仓记录
                if symbol not in self._open_trades:
                    self._open_trades[symbol] = Trade(
                        symbol=symbol, entry_date=date, exit_date=date,
                        entry_price=exec_price, exit_price=exec_price,
                        shares=shares, pnl=0, pnl_pct=0
                    )
        
        elif action == 'sell':
            shares = min(self.positions.get(symbol, 0), int(amount / exec_price))
            if shares > 0:
                revenue = shares * exec_price * (1 - self.commission)
                self.cash += revenue
                self.positions[symbol] -= shares
                if self.positions[symbol] == 0:
                    del self.positions[symbol]
                # 平仓
                if symbol in self._open_trades:
                    t = self._open_trades.pop(symbol)
                    pnl = (exec_price - t.entry_price) * shares
                    self.trades.append(Trade(
                        symbol=symbol, entry_date=t.entry_date, exit_date=date,
                        entry_price=t.entry_price, exit_price=exec_price,
                        shares=shares, pnl=pnl,
                        pnl_pct=(exec_price / t.entry_price - 1) * 100
                    ))
    
    def update_equity(self, date: pd.Timestamp, prices: Dict[str, float]):
        """更新权益曲线"""
        position_value = sum(self.positions.get(s, 0) * prices.get(s, 0) 
                            for s in self.positions)
        total = self.cash + position_value
        self.equity_curve.append({'date': date, 'equity': total, 'cash': self.cash})
    
    def run_strategy(self, data: Dict[str, pd.DataFrame], 
                     signal_fn: Callable,
                     allocation: float = 0.3):
        """运行策略回测"""
        self.reset()
        
        # 对齐日期
        all_dates = sorted(set.intersection(*[set(d.index) for d in data.values()]))
        
        for date in all_dates:
            prices = {sym: df.loc[date, 'close'] for sym, df in data.items()}
            self.update_equity(date, prices)
            
            # 生成信号
            signals = signal_fn(data, date, prices)
            
            # 执行交易
            for sym, action in signals.items():
                alloc = self.initial_capital * allocation
                self.order(sym, action, alloc, prices[sym], date)
        
        return self.report()
    
    def run_buy_and_hold(self, data: Dict[str, pd.DataFrame], symbol: str) -> dict:
        """买入持有基准收益"""
        df = data[symbol]
        start_price = df['close'].iloc[0]
        end_price = df['close'].iloc[-1]
        shares = int(self.initial_capital / start_price)
        final_value = shares * end_price
        return {
            'strategy': 'Buy & Hold',
            'total_return_pct': round((final_value / self.initial_capital - 1) * 100, 2),
            'final_value': round(final_value, 2)
        }
    
    def report(self) -> dict:
        """回测报告"""
        if not self.equity_curve:
            return {}
        
        eq = pd.DataFrame(self.equity_curve).set_index('date')
        eq['returns'] = eq['equity'].pct_change()
        
        final_equity = eq['equity'].iloc[-1]
        total_return = (final_equity / self.initial_capital - 1) * 100
        
        # 年化收益
        days = len(eq)
        annual_return = ((final_equity / self.initial_capital) ** (252 / days) - 1) * 100
        
        # Sharpe
        sharpe = eq['returns'].mean() / eq['returns'].std() * np.sqrt(252)
        
        # 最大回撤
        peak = eq['equity'].expanding().max()
        drawdown = (eq['equity'] / peak - 1) * 100
        max_dd = drawdown.min()
        
        # 胜率
        winning_trades = [t for t in self.trades if t.pnl > 0]
        win_rate = len(winning_trades) / len(self.trades) * 100 if self.trades else 0
        
        avg_pnl = np.mean([t.pnl for t in self.trades]) if self.trades else 0
        
        report = {
            'total_return_pct': round(total_return, 2),
            'annual_return_pct': round(annual_return, 2),
            'sharpe_ratio': round(sharpe, 2),
            'max_drawdown_pct': round(max_dd, 2),
            'win_rate_pct': round(win_rate, 2),
            'total_trades': len(self.trades),
            'avg_pnl': round(avg_pnl, 2),
            'final_equity': round(final_equity, 2),
        }
        
        self._print_report(report, eq, drawdown)
        return report
    
    def _print_report(self, report: dict, eq: pd.DataFrame, drawdown: pd.Series):
        print("\n" + "=" * 50)
        print("  回测报告")
        print("=" * 50)
        for k, v in report.items():
            print(f"  {k}: {v}")
        print("=" * 50)


# --- 示例策略: 双均线信号 ---
def ma_cross_signal(data: Dict[str, pd.DataFrame], date, prices) -> dict:
    """双均线交叉信号生成器"""
    signals = {}
    for sym, df in data.items():
        hist = df.loc[:date]
        if len(hist) < 25:
            continue
        ma5 = hist['close'].rolling(5).mean().iloc[-1]
        ma20 = hist['close'].rolling(20).mean().iloc[-1]
        ma5_prev = hist['close'].rolling(5).mean().iloc[-2]
        ma20_prev = hist['close'].rolling(20).mean().iloc[-2]
        
        # 金叉买入
        if ma5_prev <= ma20_prev and ma5 > ma20:
            signals[sym] = 'buy'
        # 死叉卖出
        elif ma5_prev >= ma20_prev and ma5 < ma20:
            signals[sym] = 'sell'
    return signals

if __name__ == "__main__":
    from data.fetch_data import fetch_a_stock
    
    print("加载数据...")
    stocks = {'000001': fetch_a_stock("000001", start="20220101"),
              '000002': fetch_a_stock("000002", start="20220101")}
    
    engine = BacktestEngine(initial_capital=100000, commission=0.0003, slippage=0.001)
    
    print("运行双均线回测...")
    result = engine.run_strategy(stocks, ma_cross_signal, allocation=0.5)
    
    # 基准对比
    bh = engine.run_buy_and_hold(stocks, '000001')
    print(f"\n基准 (Buy & Hold 000001): {bh['total_return_pct']}%")
