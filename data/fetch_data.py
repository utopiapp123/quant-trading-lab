"""
数据获取模块 — 支持 A 股、美股、加密货币
"""
import akshare as ak
import pandas as pd

def fetch_a_stock(symbol: str, start: str = "20200101", end: str = "20251231") -> pd.DataFrame:
    """获取 A 股日线数据"""
    df = ak.stock_zh_a_hist(symbol=symbol, period="daily", 
                            start_date=start, end_date=end, adjust="qfq")
    df["date"] = pd.to_datetime(df["日期"])
    df = df.rename(columns={"开盘": "open", "收盘": "close", "最高": "high", 
                            "最低": "low", "成交量": "volume"})
    return df.set_index("date")[["open", "high", "low", "close", "volume"]].dropna()

def fetch_index(symbol: str = "000300", start: str = "20200101", end: str = "20251231") -> pd.DataFrame:
    """获取指数日线 (沪深300)"""
    df = ak.stock_zh_index_daily(symbol=f"sh{symbol}")
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").dropna()

if __name__ == "__main__":
    df = fetch_a_stock("000001")
    print(df.tail())
    print(f"\n共 {len(df)} 条数据")
