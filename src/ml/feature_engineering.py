"""
特征工程模块
从历史K线数据提取技术指标作为ML特征
"""

import pandas as pd
import numpy as np
from typing import Optional


class FeatureEngineering:
    """技术指标特征工程"""
    
    def __init__(self, lookback_days: int = 120):
        """
        Args:
            lookback_days: 历史数据回溯天数（用于计算长周期指标）
        """
        self.lookback_days = lookback_days
    
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有技术指标特征
        
        Args:
            df: 包含OHLCV数据的DataFrame
                必须包含列: open, high, low, close, volume
        
        Returns:
            添加了技术指标列的DataFrame
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        # 1. 移动平均线
        df = self._add_moving_averages(df)
        
        # 2. RSI
        df = self._add_rsi(df)
        
        # 3. MACD
        df = self._add_macd(df)
        
        # 4. 布林带
        df = self._add_bollinger_bands(df)
        
        # 5. 成交量特征
        df = self._add_volume_features(df)
        
        # 6. 价格变化率
        df = self._add_price_changes(df)
        
        # 7. 波动率
        df = self._add_volatility(df)
        
        # 8. K线形态特征
        df = self._add_candlestick_features(df)
        
        return df
    
    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """移动平均线"""
        for period in [5, 10, 20, 60]:
            df[f'ma_{period}'] = df['close'].rolling(window=period).mean()
            # 价格与MA的偏离度
            df[f'close_ma{period}_ratio'] = df['close'] / df[f'ma_{period}']
        
        # MA交叉信号
        df['ma5_ma10_cross'] = (df['ma_5'] > df['ma_10']).astype(int)
        df['ma10_ma20_cross'] = (df['ma_10'] > df['ma_20']).astype(int)
        
        return df
    
    def _add_rsi(self, df: pd.DataFrame, periods: list = [6, 12, 24]) -> pd.DataFrame:
        """RSI相对强弱指数"""
        delta = df['close'].diff()
        
        for period in periods:
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            
            rs = avg_gain / (avg_loss + 1e-10)
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        return df
    
    def _add_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """MACD指标"""
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # MACD与价格的比率
        df['macd_price_ratio'] = df['macd'] / (df['close'] + 1e-10)
        
        return df
    
    def _add_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std_dev: int = 2) -> pd.DataFrame:
        """布林带"""
        ma = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        
        df['bb_upper'] = ma + (std * std_dev)
        df['bb_lower'] = ma - (std * std_dev)
        df['bb_middle'] = ma
        
        # 价格在布林带中的位置（0-1）
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
        
        # 布林带宽度
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['bb_middle'] + 1e-10)
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """成交量特征"""
        # 成交量移动平均
        for period in [5, 20]:
            df[f'volume_ma_{period}'] = df['volume'].rolling(window=period).mean()
        
        # 成交量比率
        df['volume_ratio_5'] = df['volume'] / (df['volume_ma_5'] + 1e-10)
        df['volume_ratio_20'] = df['volume'] / (df['volume_ma_20'] + 1e-10)
        
        # 成交量变化率
        df['volume_change'] = df['volume'].pct_change()
        
        # 量价关系
        df['price_volume_corr_5'] = df['close'].rolling(5).corr(df['volume'])
        
        return df
    
    def _add_price_changes(self, df: pd.DataFrame) -> pd.DataFrame:
        """价格变化率"""
        for period in [1, 3, 5, 10, 20]:
            df[f'return_{period}d'] = df['close'].pct_change(periods=period)
        
        # 高低价变化
        df['high_low_range'] = (df['high'] - df['low']) / (df['close'] + 1e-10)
        df['open_close_range'] = (df['close'] - df['open']) / (df['open'] + 1e-10)
        
        return df
    
    def _add_volatility(self, df: pd.DataFrame, periods: list = [5, 10, 20]) -> pd.DataFrame:
        """波动率"""
        returns = df['close'].pct_change()
        
        for period in periods:
            df[f'volatility_{period}d'] = returns.rolling(window=period).std()
        
        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        df['atr_14'] = true_range.rolling(window=14).mean()
        df['atr_ratio'] = df['atr_14'] / (df['close'] + 1e-10)
        
        return df
    
    def _add_candlestick_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """K线形态特征"""
        # 上影线比例
        df['upper_shadow'] = (df['high'] - np.maximum(df['open'], df['close'])) / (df['high'] - df['low'] + 1e-10)
        
        # 下影线比例
        df['lower_shadow'] = (np.minimum(df['open'], df['close']) - df['low']) / (df['high'] - df['low'] + 1e-10)
        
        # 实体比例
        df['body_ratio'] = np.abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-10)
        
        # 是否阳线
        df['is_bullish'] = (df['close'] > df['open']).astype(int)
        
        return df
    
    def get_feature_names(self) -> list:
        """获取所有特征名称"""
        features = []
        
        # MA相关
        for period in [5, 10, 20, 60]:
            features.extend([f'ma_{period}', f'close_ma{period}_ratio'])
        features.extend(['ma5_ma10_cross', 'ma10_ma20_cross'])
        
        # RSI
        features.extend(['rsi_6', 'rsi_12', 'rsi_24'])
        
        # MACD
        features.extend(['macd', 'macd_signal', 'macd_hist', 'macd_price_ratio'])
        
        # 布林带
        features.extend(['bb_upper', 'bb_lower', 'bb_middle', 'bb_position', 'bb_width'])
        
        # 成交量
        features.extend(['volume_ma_5', 'volume_ma_20', 'volume_ratio_5', 'volume_ratio_20', 
                        'volume_change', 'price_volume_corr_5'])
        
        # 价格变化
        features.extend(['return_1d', 'return_3d', 'return_5d', 'return_10d', 'return_20d',
                        'high_low_range', 'open_close_range'])
        
        # 波动率
        features.extend(['volatility_5d', 'volatility_10d', 'volatility_20d', 'atr_14', 'atr_ratio'])
        
        # K线形态
        features.extend(['upper_shadow', 'lower_shadow', 'body_ratio', 'is_bullish'])
        
        return features
