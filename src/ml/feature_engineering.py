"""
特征工程模块
从历史K线数据提取技术指标作为ML特征
"""

import pandas as pd
import numpy as np
from typing import Optional


class FeatureEngineering:
    """技术指标特征工程"""
    
    def __init__(self, lookback_days: int = 250):
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
        
        # 9. 换手率特征
        df = self._add_turnover_features(df)
        
        # 10. 估值特征
        df = self._add_valuation_features(df)
        
        # 11. 动量特征
        df = self._add_momentum_features(df)
        
        # 12. 市场微观结构特征
        df = self._add_microstructure_features(df)
        
        # 13. 时序模式特征
        df = self._add_temporal_features(df)
        
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
    
    def _add_turnover_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """换手率特征"""
        if 'turn' not in df.columns:
            return df
        
        # 换手率移动平均
        for period in [5, 10, 20]:
            df[f'turn_ma_{period}'] = df['turn'].rolling(window=period).mean()
        
        # 换手率变化率
        df['turn_change'] = df['turn'].pct_change()
        
        # 换手率与成交量的关系
        df['turn_volume_ratio'] = df['turn'] / (df['volume'] + 1e-10)
        
        # 换手率与价格的关系
        df['turn_price_corr_5'] = df['turn'].rolling(5).corr(df['close'])
        
        return df
    
    def _add_valuation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """估值特征"""
        # 市盈率特征
        if 'peTTM' in df.columns:
            df['pe_ma_20'] = df['peTTM'].rolling(window=20).mean()
            df['pe_ratio'] = df['peTTM'] / (df['pe_ma_20'] + 1e-10)
        
        # 市净率特征
        if 'pbMRQ' in df.columns:
            df['pb_ma_20'] = df['pbMRQ'].rolling(window=20).mean()
            df['pb_ratio'] = df['pbMRQ'] / (df['pb_ma_20'] + 1e-10)
        
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """动量特征 - 捕捉价格趋势的强度和方向"""
        returns = df['close'].pct_change()
        
        # 多周期动量
        for period in [5, 10, 20, 60]:
            df[f'momentum_{period}d'] = df['close'] / df['close'].shift(period) - 1
            df[f'roc_{period}d'] = returns.rolling(period).sum()
        
        # Williams %R (超买超卖指标)
        high_14 = df['high'].rolling(14).max()
        low_14 = df['low'].rolling(14).min()
        df['williams_r_14'] = (high_14 - df['close']) / (high_14 - low_14 + 1e-10) * -100
        
        # CCI (商品通道指数)
        tp = (df['high'] + df['low'] + df['close']) / 3
        tp_ma = tp.rolling(20).mean()
        tp_md = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        df['cci_20'] = (tp - tp_ma) / (0.015 * tp_md + 1e-10)
        
        # MFI (资金流量指数) - 结合成交量的RSI
        tp_change = tp.diff()
        pos_flow = tp_change.where(tp_change > 0, 0) * df['volume']
        neg_flow = -tp_change.where(tp_change < 0, 0) * df['volume']
        pos_flow_ma = pos_flow.rolling(14).sum()
        neg_flow_ma = neg_flow.rolling(14).sum()
        mfr = pos_flow_ma / (neg_flow_ma + 1e-10)
        df['mfi_14'] = 100 - (100 / (1 + mfr))
        
        # 随机指标 (Stochastic Oscillator)
        low_14 = df['low'].rolling(14).min()
        high_14 = df['high'].rolling(14).max()
        df['stoch_k'] = (df['close'] - low_14) / (high_14 - low_14 + 1e-10) * 100
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        
        # ADX (平均趋向指数) - 趋势强度
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        tr = self._compute_true_range(df)
        atr_14 = tr.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / (atr_14 + 1e-10))
        minus_di = 100 * (minus_dm.rolling(14).mean() / (atr_14 + 1e-10))
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        df['adx_14'] = dx.rolling(14).mean()
        
        # 动量加速度/减速度
        mom_5 = df['momentum_5d']
        df['momentum_accel'] = mom_5.diff(5)  # 动量变化率
        df['momentum_decel'] = -df['momentum_accel']
        
        return df
    
    def _compute_true_range(self, df: pd.DataFrame) -> pd.Series:
        """计算真实波幅"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        return np.maximum(high_low, np.maximum(high_close, low_close))
    
    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """市场微观结构特征 - 捕捉流动性、价格冲击等"""
        returns = df['close'].pct_change()
        
        # Amihud非流动性 (|return| / volume) - 流动性越低，值越大
        df['amihud_illiquidity_20'] = (np.abs(returns) / (df['volume'] + 1e-10)).rolling(20).mean()
        
        # Kyle's Lambda代理 - 价格冲击（回归斜率）
        def rolling_beta(y, x, window):
            result = []
            for i in range(len(y)):
                if i < window:
                    result.append(np.nan)
                else:
                    y_slice = y.iloc[i-window:i]
                    x_slice = x.iloc[i-window:i]
                    if x_slice.std() < 1e-10:
                        result.append(0)
                    else:
                        result.append(np.cov(y_slice, x_slice)[0,1] / (x_slice.var() + 1e-10))
            return pd.Series(result, index=y.index)
        
        # 用累计成交量作为自变量，价格作为因变量
        cum_vol = df['volume'].cumsum()
        df['kyle_lambda_proxy'] = rolling_beta(df['close'], cum_vol, 20)
        
        # 买卖价差代理 (Roll模型)
        df['bid_ask_spread_proxy'] = 2 * np.sqrt(np.maximum(
            -returns.rolling(20).cov(returns.shift(1)), 0))
        
        # 成交量波动
        df['volume_std_20'] = df['volume'].rolling(20).std() / (df['volume'].rolling(20).mean() + 1e-10)
        
        # 高低价成交量比
        up_vol = df['volume'].where(returns > 0, 0).rolling(20).sum()
        down_vol = df['volume'].where(returns < 0, 0).rolling(20).sum()
        df['high_low_vol_ratio'] = up_vol / (down_vol + 1e-10)
        
        # 日内强度 (Close Location Value)
        df['intraday_intensity'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
        
        # OBV (能量潮)
        obv_direction = np.sign(returns)
        df['obv'] = (obv_direction * df['volume']).cumsum()
        df['obv_ma_20'] = df['obv'].rolling(20).mean()
        df['obv_slope'] = df['obv'].diff(5) / (df['obv_ma_20'] + 1e-10)
        
        # VWAP偏离度
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).rolling(20).sum() / (df['volume'].rolling(20).sum() + 1e-10)
        df['vwap_deviation'] = (df['close'] - vwap) / (vwap + 1e-10)
        
        # 价格加速度
        df['price_acceleration'] = returns.diff()
        
        # 跳空频率
        gap = df['open'] - df['close'].shift(1)
        gap_pct = gap / (df['close'].shift(1) + 1e-10)
        df['gap_up_freq_20'] = (gap_pct > 0.01).rolling(20).mean()
        df['gap_down_freq_20'] = (gap_pct < -0.01).rolling(20).mean()
        
        # 距离涨跌停的距离（A股10%限制）
        df['limit_up_proximity'] = (df['close'] * 1.1 - df['close']) / (df['close'] + 1e-10)
        df['limit_down_proximity'] = (df['close'] - df['close'] * 0.9) / (df['close'] + 1e-10)
        
        return df
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """时序模式特征 - 捕捉日历效应、趋势持续性等"""
        returns = df['close'].pct_change()
        
        # 日历特征（如果有日期索引）
        if hasattr(df.index, 'dayofweek'):
            df['day_of_week'] = df.index.dayofweek
            df['month_of_year'] = df.index.month
            df['week_of_month'] = (df.index.day - 1) // 7 + 1
        else:
            # 假设是连续交易日，用模运算近似
            n = len(df)
            df['day_of_week'] = np.arange(n) % 5
            df['month_of_year'] = (np.arange(n) % 22) // 5 + 1
            df['week_of_month'] = (np.arange(n) % 22) // 7 + 1
        
        # 距离20日高/低点的天数
        rolling_high_idx = df['high'].rolling(20).apply(lambda x: x.argmax(), raw=True)
        rolling_low_idx = df['low'].rolling(20).apply(lambda x: x.argmin(), raw=True)
        df['days_since_high_20'] = 20 - rolling_high_idx
        df['days_since_low_20'] = 20 - rolling_low_idx
        
        # 连续涨/跌天数
        is_up = (returns > 0).astype(int)
        is_down = (returns < 0).astype(int)
        
        def consecutive_count(series):
            result = []
            count = 0
            for val in series:
                if val == 1:
                    count += 1
                else:
                    count = 0
                result.append(count)
            return pd.Series(result, index=series.index)
        
        df['consecutive_up_days'] = consecutive_count(is_up)
        df['consecutive_down_days'] = consecutive_count(is_down)
        
        # 趋势强度 (线性回归R²)
        def trend_r2(y, window):
            result = []
            for i in range(len(y)):
                if i < window:
                    result.append(np.nan)
                else:
                    y_slice = y.iloc[i-window:i].values
                    x = np.arange(window)
                    if np.std(y_slice) < 1e-10:
                        result.append(0)
                    else:
                        corr = np.corrcoef(x, y_slice)[0, 1]
                        result.append(corr ** 2 if not np.isnan(corr) else 0)
            return pd.Series(result, index=y.index)
        
        df['trend_strength_20'] = trend_r2(df['close'], 20)
        df['trend_strength_60'] = trend_r2(df['close'], 60)
        
        # 均值回归得分 (当前价格偏离均线的程度)
        ma_20 = df['close'].rolling(20).mean()
        std_20 = df['close'].rolling(20).std()
        df['mean_reversion_score'] = (df['close'] - ma_20) / (std_20 + 1e-10)
        
        # 自相关性
        df['autocorr_1'] = returns.rolling(20).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 1 else 0, raw=False)
        df['autocorr_5'] = returns.rolling(20).apply(
            lambda x: x.autocorr(lag=5) if len(x) > 5 else 0, raw=False)
        
        # Hurst指数代理 (R/S分析简化版)
        def hurst_proxy(y, window):
            result = []
            for i in range(len(y)):
                if i < window:
                    result.append(np.nan)
                else:
                    y_slice = y.iloc[i-window:i].values
                    mean_y = np.mean(y_slice)
                    dev_cumsum = np.cumsum(y_slice - mean_y)
                    r = np.max(dev_cumsum) - np.min(dev_cumsum)
                    s = np.std(y_slice)
                    if s < 1e-10:
                        result.append(0.5)
                    else:
                        rs = r / s
                        # H ≈ log(R/S) / log(n)
                        h = np.log(rs + 1e-10) / np.log(window)
                        result.append(np.clip(h, 0, 1))
            return pd.Series(result, index=y.index)
        
        df['hurst_exponent_proxy'] = hurst_proxy(returns, 60)
        
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
        
        # 换手率
        features.extend(['turn_ma_5', 'turn_ma_10', 'turn_ma_20', 'turn_change', 
                        'turn_volume_ratio', 'turn_price_corr_5'])
        
        # 估值
        features.extend(['pe_ma_20', 'pe_ratio', 'pb_ma_20', 'pb_ratio'])
        
        # 动量特征
        features.extend(['momentum_5d', 'momentum_10d', 'momentum_20d', 'momentum_60d',
                        'roc_5d', 'roc_10d', 'roc_20d', 'roc_60d',
                        'williams_r_14', 'cci_20', 'mfi_14',
                        'stoch_k', 'stoch_d', 'adx_14',
                        'momentum_accel', 'momentum_decel'])
        
        # 市场微观结构特征
        features.extend(['amihud_illiquidity_20', 'kyle_lambda_proxy',
                        'bid_ask_spread_proxy', 'volume_std_20',
                        'high_low_vol_ratio', 'intraday_intensity',
                        'obv', 'obv_ma_20', 'obv_slope',
                        'vwap_deviation', 'price_acceleration',
                        'gap_up_freq_20', 'gap_down_freq_20',
                        'limit_up_proximity', 'limit_down_proximity'])
        
        # 时序模式特征
        features.extend(['day_of_week', 'month_of_year', 'week_of_month',
                        'days_since_high_20', 'days_since_low_20',
                        'consecutive_up_days', 'consecutive_down_days',
                        'trend_strength_20', 'trend_strength_60',
                        'mean_reversion_score', 'autocorr_1', 'autocorr_5',
                        'hurst_exponent_proxy'])
        
        return features
