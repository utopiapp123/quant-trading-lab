"""
LSTM 股价预测 | LSTM Stock Price Predictor
基于 LSTM 神经网络预测次日涨跌方向
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

class LSTMPredictor:
    """LSTM 股价涨跌预测器"""
    
    def __init__(self, seq_length: int = 30, epochs: int = 50, batch_size: int = 32):
        self.seq_length = seq_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = MinMaxScaler()
        self.model = None
        
    def _create_sequences(self, data: np.ndarray) -> tuple:
        """构建时间序列样本"""
        X, y = [], []
        for i in range(self.seq_length, len(data)):
            X.append(data[i-self.seq_length:i])
            y.append(1 if data[i] > data[i-1] else 0)  # 涨:1, 跌:0
        return np.array(X), np.array(y)
    
    def _build_model(self, input_shape: tuple):
        """构建 LSTM 模型"""
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model = model
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """特征工程"""
        df = df.copy()
        # 价格特征
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
        # 技术指标
        df['ma5'] = df['close'].rolling(5).mean() / df['close']
        df['ma10'] = df['close'].rolling(10).mean() / df['close']
        df['ma20'] = df['close'].rolling(20).mean() / df['close']
        df['volatility'] = df['returns'].rolling(20).std()
        df['volume_ma5'] = df['volume'].rolling(5).mean() / df['volume']
        
        # 动量
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        
        feature_cols = ['returns', 'log_returns', 'high_low_ratio', 'close_open_ratio',
                        'ma5', 'ma10', 'ma20', 'volatility', 'volume_ma5', 
                        'momentum_5', 'momentum_10']
        return df[feature_cols].dropna()
    
    def fit(self, df: pd.DataFrame, test_size: float = 0.2):
        """训练模型"""
        features = self.prepare_features(df)
        scaled = self.scaler.fit_transform(features)
        
        X, y = self._create_sequences(scaled)
        split = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        print(f"训练集: {len(X_train)} 样本, 测试集: {len(X_test)} 样本")
        print(f"特征数: {X.shape[2]}, 序列长度: {self.seq_length}")
        
        self._build_model((X.shape[1], X.shape[2]))
        
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size,
                       validation_data=(X_test, y_test), callbacks=[early_stop], verbose=0)
        
        # 评估
        loss, acc = self.model.evaluate(X_test, y_test, verbose=0)
        y_pred = (self.model.predict(X_test, verbose=0) > 0.5).astype(int)
        
        return {
            'accuracy': round(acc * 100, 2),
            'test_samples': len(X_test),
            'features': X.shape[2]
        }, y_test, y_pred
    
    def predict_next(self, df: pd.DataFrame) -> str:
        """预测下一个交易日的涨跌"""
        features = self.prepare_features(df)
        scaled = self.scaler.transform(features)
        latest = scaled[-self.seq_length:].reshape(1, self.seq_length, -1)
        prob = self.model.predict(latest, verbose=0)[0][0]
        direction = "📈 看涨" if prob > 0.5 else "📉 看跌"
        return f"{direction} (置信度: {max(prob, 1-prob):.1%})"

if __name__ == "__main__":
    from data.fetch_data import fetch_a_stock
    
    print("正在获取数据...")
    df = fetch_a_stock("000001", start="20200101", end="20251231")
    print(f"数据: {len(df)} 条\n")
    
    model = LSTMPredictor(seq_length=30, epochs=50)
    results, y_true, y_pred = model.fit(df)
    
    print(f"\n======= 模型评估 =======")
    print(f"准确率: {results['accuracy']}%")
    print(f"测试样本: {results['test_samples']}")
    print(classification_report(y_true, y_pred, target_names=['跌', '涨']))
    
    print(f"\n明日预测: {model.predict_next(df)}")
