"""
高级特征工程模块
包含更多技术指标、市场情绪、行业相关性等特征
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class AdvancedFeatureEngineer:
    """高级特征工程类"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def create_all_features(self, df: pd.DataFrame, stock_code: str = None) -> pd.DataFrame:
        """创建所有特征"""
        df = df.copy()
        
        # 1. 基础技术指标
        df = self._add_technical_indicators(df)
        
        # 2. 价格模式特征
        df = self._add_price_patterns(df)
        
        # 3. 成交量特征
        df = self._add_volume_features(df)
        
        # 4. 波动率特征
        df = self._add_volatility_features(df)
        
        # 5. 时间序列特征
        df = self._add_time_features(df)
        
        # 6. 统计特征
        df = self._add_statistical_features(df)
        
        # 7. 动量特征
        df = self._add_momentum_features(df)
        
        # 8. 趋势强度特征
        df = self._add_trend_strength(df)
        
        # 9. 支撑阻力特征
        df = self._add_support_resistance(df)
        
        # 10. K线形态特征
        df = self._add_candlestick_patterns(df)
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加技术指标"""
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # 移动平均线
        for period in [5, 10, 20, 30, 60, 120]:
            df[f'ma_{period}'] = close.rolling(period).mean()
            df[f'ema_{period}'] = close.ewm(span=period).mean()
            df[f'close_ma{period}_ratio'] = close / df[f'ma_{period}']
        
        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        df['macd_hist_diff'] = df['macd_hist'].diff()
        
        # RSI
        for period in [6, 12, 24]:
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # 布林带
        for period in [20]:
            ma = close.rolling(period).mean()
            std = close.rolling(period).std()
            df[f'bb_upper_{period}'] = ma + 2 * std
            df[f'bb_lower_{period}'] = ma - 2 * std
            df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / ma
            df[f'bb_position_{period}'] = (close - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'])
        
        # KDJ
        low_min = low.rolling(9).min()
        high_max = high.rolling(9).max()
        rsv = (close - low_min) / (high_max - low_min) * 100
        df['kdj_k'] = rsv.ewm(com=2).mean()
        df['kdj_d'] = df['kdj_k'].ewm(com=2).mean()
        df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']
        
        # ATR (Average True Range)
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr_14'] = true_range.rolling(14).mean()
        df['atr_ratio'] = df['atr_14'] / close
        
        # OBV (On Balance Volume)
        df['obv'] = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        df['obv_ma5'] = df['obv'].rolling(5).mean()
        df['obv_ma20'] = df['obv'].rolling(20).mean()
        
        # VWAP (Volume Weighted Average Price)
        df['vwap'] = (volume * close).rolling(20).sum() / volume.rolling(20).sum()
        df['close_vwap_ratio'] = close / df['vwap']
        
        return df
    
    def _add_price_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加价格模式特征"""
        close = df['close']
        high = df['high']
        low = df['low']
        open_price = df['open']
        
        # 价格变化率
        for period in [1, 3, 5, 10, 20]:
            df[f'return_{period}d'] = close.pct_change(period)
        
        # 日内波动
        df['intraday_return'] = (close - open_price) / open_price
        df['upper_shadow'] = (high - np.maximum(open_price, close)) / (high - low + 1e-10)
        df['lower_shadow'] = (np.minimum(open_price, close) - low) / (high - low + 1e-10)
        df['body_ratio'] = abs(close - open_price) / (high - low + 1e-10)
        
        # 连续涨跌天数
        df['up'] = (close > close.shift(1)).astype(int)
        df['consecutive_up'] = df['up'].groupby((df['up'] != df['up'].shift()).cumsum()).cumsum()
        df['consecutive_down'] = (1 - df['up']).groupby(((1 - df['up']) != (1 - df['up']).shift()).cumsum()).cumsum()
        
        # 价格位置
        df['price_position_20d'] = (close - close.rolling(20).min()) / (close.rolling(20).max() - close.rolling(20).min() + 1e-10)
        df['price_position_60d'] = (close - close.rolling(60).min()) / (close.rolling(60).max() - close.rolling(60).min() + 1e-10)
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加成交量特征"""
        volume = df['volume']
        close = df['close']
        
        # 成交量移动平均
        for period in [5, 10, 20]:
            df[f'volume_ma_{period}'] = volume.rolling(period).mean()
            df[f'volume_ratio_{period}'] = volume / df[f'volume_ma_{period}']
        
        # 成交量变化率
        df['volume_change_1d'] = volume.pct_change()
        df['volume_change_5d'] = volume.pct_change(5)
        
        # 量价相关性
        df['price_volume_corr_5'] = close.rolling(5).corr(volume)
        df['price_volume_corr_20'] = close.rolling(20).corr(volume)
        
        # 成交量波动率
        df['volume_volatility_5'] = volume.rolling(5).std() / volume.rolling(5).mean()
        df['volume_volatility_20'] = volume.rolling(20).std() / volume.rolling(20).mean()
        
        # 异常成交量
        df['volume_zscore'] = (volume - volume.rolling(20).mean()) / volume.rolling(20).std()
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加波动率特征"""
        close = df['close']
        returns = close.pct_change()
        
        # 历史波动率
        for period in [5, 10, 20, 60]:
            df[f'volatility_{period}d'] = returns.rolling(period).std() * np.sqrt(252)
        
        # Parkinson波动率（使用最高价和最低价）
        high = df['high']
        low = df['low']
        parkinson_var = (np.log(high / low) ** 2) / (4 * np.log(2))
        df['parkinson_volatility_20'] = np.sqrt(252 * parkinson_var.rolling(20).mean())
        
        # Garman-Klass波动率
        open_price = df['open']
        gk_var = 0.5 * np.log(high / low) ** 2 - (2 * np.log(2) - 1) * np.log(close / open_price) ** 2
        df['gk_volatility_20'] = np.sqrt(252 * gk_var.rolling(20).mean())
        
        # 波动率比率
        df['volatility_ratio_5_20'] = df['volatility_5d'] / (df['volatility_20d'] + 1e-10)
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加时间特征"""
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df['dayofweek'] = df['date'].dt.dayofweek
            df['month'] = df['date'].dt.month
            df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
            df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
            df['is_quarter_end'] = df['date'].dt.is_quarter_end.astype(int)
            
            # 周期性编码
            df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
            df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加统计特征"""
        close = df['close']
        returns = close.pct_change()
        
        # 偏度和峰度
        df['return_skew_20'] = returns.rolling(20).skew()
        df['return_kurtosis_20'] = returns.rolling(20).kurt()
        
        # 自相关性
        df['return_autocorr_5'] = returns.rolling(20).apply(lambda x: x.autocorr(lag=5), raw=False)
        df['return_autocorr_10'] = returns.rolling(20).apply(lambda x: x.autocorr(lag=10), raw=False)
        
        # Hurst指数（简化版）
        def hurst_exponent(ts, max_lag=20):
            lags = range(2, max_lag)
            tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
        
        df['hurst_60'] = close.rolling(60).apply(hurst_exponent, raw=True)
        
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加动量特征"""
        close = df['close']
        
        # ROC (Rate of Change)
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = (close - close.shift(period)) / close.shift(period) * 100
        
        # Williams %R
        high = df['high']
        low = df['low']
        for period in [14]:
            high_max = high.rolling(period).max()
            low_min = low.rolling(period).min()
            df[f'williams_r_{period}'] = (high_max - close) / (high_max - low_min) * -100
        
        # CCI (Commodity Channel Index)
        tp = (df['high'] + df['low'] + close) / 3
        df['cci_20'] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())
        
        # MFI (Money Flow Index)
        tp = (df['high'] + df['low'] + close) / 3
        mf = tp * df['volume']
        pos_mf = mf.where(tp > tp.shift(1), 0).rolling(14).sum()
        neg_mf = mf.where(tp < tp.shift(1), 0).rolling(14).sum()
        df['mfi_14'] = 100 - (100 / (1 + pos_mf / (neg_mf + 1e-10)))
        
        return df
    
    def _add_trend_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加趋势强度特征"""
        close = df['close']
        
        # ADX (Average Directional Index)
        high = df['high']
        low = df['low']
        
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = pd.concat([
            high - low,
            abs(high - close.shift()),
            abs(low - close.shift())
        ], axis=1).max(axis=1)
        
        atr_14 = tr.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr_14)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr_14)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        df['adx_14'] = dx.rolling(14).mean()
        df['plus_di_14'] = plus_di
        df['minus_di_14'] = minus_di
        
        # 线性回归斜率
        def linear_slope(x):
            if len(x) < 2:
                return 0
            y = np.arange(len(x))
            slope = np.polyfit(y, x, 1)[0]
            return slope
        
        df['trend_slope_20'] = close.rolling(20).apply(linear_slope, raw=True)
        df['trend_slope_60'] = close.rolling(60).apply(linear_slope, raw=True)
        
        # R² (趋势拟合度)
        def r_squared(x):
            if len(x) < 2:
                return 0
            y = np.arange(len(x))
            slope, intercept = np.polyfit(y, x, 1)
            y_pred = slope * y + intercept
            ss_res = np.sum((x - y_pred) ** 2)
            ss_tot = np.sum((x - np.mean(x)) ** 2)
            return 1 - (ss_res / (ss_tot + 1e-10))
        
        df['trend_r2_20'] = close.rolling(20).apply(r_squared, raw=True)
        
        return df
    
    def _add_support_resistance(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加支撑阻力特征"""
        close = df['close']
        high = df['high']
        low = df['low']
        
        # 枢轴点
        pivot = (high + low + close) / 3
        df['pivot'] = pivot
        df['r1'] = 2 * pivot - low
        df['s1'] = 2 * pivot - high
        df['r2'] = pivot + (high - low)
        df['s2'] = pivot - (high - low)
        
        # 距离枢轴点的距离
        df['dist_to_pivot'] = (close - pivot) / pivot
        df['dist_to_r1'] = (df['r1'] - close) / close
        df['dist_to_s1'] = (close - df['s1']) / close
        
        # 近期高低点
        df['recent_high_20'] = high.rolling(20).max()
        df['recent_low_20'] = low.rolling(20).min()
        df['dist_to_recent_high'] = (df['recent_high_20'] - close) / close
        df['dist_to_recent_low'] = (close - df['recent_low_20']) / close
        
        return df
    
    def _add_candlestick_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加K线形态特征"""
        open_price = df['open']
        close = df['close']
        high = df['high']
        low = df['low']
        
        body = abs(close - open_price)
        upper_shadow = high - np.maximum(open_price, close)
        lower_shadow = np.minimum(open_price, close) - low
        candle_range = high - low
        
        # 十字星
        df['doji'] = (body < candle_range * 0.1).astype(int)
        
        # 锤子线
        df['hammer'] = ((lower_shadow > 2 * body) & (upper_shadow < body * 0.3)).astype(int)
        
        # 上吊线
        df['hanging_man'] = ((upper_shadow > 2 * body) & (lower_shadow < body * 0.3)).astype(int)
        
        # 大阳线
        df['big_bullish'] = ((close > open_price) & (body > candle_range * 0.7)).astype(int)
        
        # 大阴线
        df['big_bearish'] = ((close < open_price) & (body > candle_range * 0.7)).astype(int)
        
        # 吞没形态
        df['bullish_engulfing'] = ((close > open_price) & 
                                   (close.shift(1) < open_price.shift(1)) &
                                   (close > open_price.shift(1)) &
                                   (open_price < close.shift(1))).astype(int)
        
        df['bearish_engulfing'] = ((close < open_price) & 
                                   (close.shift(1) > open_price.shift(1)) &
                                   (close < open_price.shift(1)) &
                                   (open_price > close.shift(1))).astype(int)
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """获取所有特征名称"""
        features = []
        
        # 技术指标
        for period in [5, 10, 20, 30, 60, 120]:
            features.extend([f'ma_{period}', f'ema_{period}', f'close_ma{period}_ratio'])
        features.extend(['macd', 'macd_signal', 'macd_hist', 'macd_hist_diff'])
        features.extend(['rsi_6', 'rsi_12', 'rsi_24'])
        features.extend(['bb_width_20', 'bb_position_20'])
        features.extend(['kdj_k', 'kdj_d', 'kdj_j'])
        features.extend(['atr_14', 'atr_ratio'])
        features.extend(['obv', 'obv_ma5', 'obv_ma20'])
        features.extend(['vwap', 'close_vwap_ratio'])
        
        # 价格模式
        features.extend(['return_1d', 'return_3d', 'return_5d', 'return_10d', 'return_20d'])
        features.extend(['intraday_return', 'upper_shadow', 'lower_shadow', 'body_ratio'])
        features.extend(['consecutive_up', 'consecutive_down'])
        features.extend(['price_position_20d', 'price_position_60d'])
        
        # 成交量
        features.extend(['volume_ratio_5', 'volume_ratio_10', 'volume_ratio_20'])
        features.extend(['volume_change_1d', 'volume_change_5d'])
        features.extend(['price_volume_corr_5', 'price_volume_corr_20'])
        features.extend(['volume_volatility_5', 'volume_volatility_20'])
        features.extend(['volume_zscore'])
        
        # 波动率
        features.extend(['volatility_5d', 'volatility_10d', 'volatility_20d', 'volatility_60d'])
        features.extend(['parkinson_volatility_20', 'gk_volatility_20'])
        features.extend(['volatility_ratio_5_20'])
        
        # 时间
        features.extend(['dayofweek', 'month', 'is_month_end', 'is_month_start', 'is_quarter_end'])
        features.extend(['dayofweek_sin', 'dayofweek_cos', 'month_sin', 'month_cos'])
        
        # 统计
        features.extend(['return_skew_20', 'return_kurtosis_20'])
        features.extend(['return_autocorr_5', 'return_autocorr_10'])
        features.extend(['hurst_60'])
        
        # 动量
        features.extend(['roc_5', 'roc_10', 'roc_20'])
        features.extend(['williams_r_14', 'cci_20', 'mfi_14'])
        
        # 趋势强度
        features.extend(['adx_14', 'plus_di_14', 'minus_di_14'])
        features.extend(['trend_slope_20', 'trend_slope_60', 'trend_r2_20'])
        
        # 支撑阻力
        features.extend(['pivot', 'r1', 's1', 'r2', 's2'])
        features.extend(['dist_to_pivot', 'dist_to_r1', 'dist_to_s1'])
        features.extend(['dist_to_recent_high', 'dist_to_recent_low'])
        
        # K线形态
        features.extend(['doji', 'hammer', 'hanging_man', 'big_bullish', 'big_bearish'])
        features.extend(['bullish_engulfing', 'bearish_engulfing'])
        
        return features
