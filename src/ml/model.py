"""
趋势预测模型
使用LightGBM预测股票未来涨跌概率
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
import logging
from datetime import datetime, timedelta

try:
    import lightgbm as lgb
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import accuracy_score, roc_auc_score
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    logging.warning("LightGBM未安装，趋势预测功能不可用")

from .feature_engineering import FeatureEngineering

logger = logging.getLogger(__name__)


class TrendPredictor:
    """趋势预测器"""
    
    def __init__(self, prediction_horizon: int = 5):
        """
        Args:
            prediction_horizon: 预测未来天数（默认5天）
        """
        self.prediction_horizon = prediction_horizon
        self.feature_eng = FeatureEngineering()
        self.model = None
        self.is_trained = False
        
        if not HAS_LGBM:
            logger.warning("LightGBM不可用，将使用规则-based预测")
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        准备训练数据
        
        Args:
            df: 包含OHLCV的历史数据
        
        Returns:
            X: 特征矩阵
            y: 标签（未来N天涨跌）
        """
        # 计算特征
        df = self.feature_eng.compute_features(df)
        
        # 创建标签：未来N天是否上涨
        df['future_return'] = df['close'].shift(-self.prediction_horizon) / df['close'] - 1
        df['label'] = (df['future_return'] > 0.02).astype(int)  # 上涨>2%为正样本
        
        # 删除NaN行
        df = df.dropna()
        
        # 获取特征列
        feature_cols = self.feature_eng.get_feature_names()
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        X = df[feature_cols]
        y = df['label']
        
        return X, y
    
    def train(self, df: pd.DataFrame, n_splits: int = 5) -> Dict:
        """
        训练模型
        
        Args:
            df: 历史数据
            n_splits: 时间序列交叉验证折数
        
        Returns:
            训练指标
        """
        if not HAS_LGBM:
            logger.warning("LightGBM不可用，跳过训练")
            return {'error': 'LightGBM not installed'}
        
        X, y = self.prepare_training_data(df)
        
        if len(X) < 100:
            logger.warning(f"训练数据不足: {len(X)} 行")
            return {'error': 'Insufficient data'}
        
        # 时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []
        
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            model = lgb.train(
                params,
                train_data,
                num_boost_round=100,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
            )
            
            y_pred = model.predict(X_val)
            auc = roc_auc_score(y_val, y_pred)
            cv_scores.append(auc)
        
        # 用全部数据训练最终模型
        full_train_data = lgb.Dataset(X, label=y)
        self.model = lgb.train(params, full_train_data, num_boost_round=100)
        self.is_trained = True
        
        metrics = {
            'cv_auc_mean': np.mean(cv_scores),
            'cv_auc_std': np.std(cv_scores),
            'cv_scores': cv_scores,
            'n_samples': len(X),
            'n_positive': int(y.sum()),
            'n_negative': int(len(y) - y.sum())
        }
        
        logger.info(f"模型训练完成: AUC={metrics['cv_auc_mean']:.3f}±{metrics['cv_auc_std']:.3f}")
        
        return metrics
    
    def predict(self, df: pd.DataFrame) -> Dict:
        """
        预测未来趋势
        
        Args:
            df: 最新的历史数据（至少需要120天）
        
        Returns:
            预测结果字典
        """
        if len(df) < 60:
            return {
                'error': '数据不足',
                'prediction': None,
                'confidence': 0
            }
        
        # 计算特征
        df_features = self.feature_eng.compute_features(df)
        
        # 取最后一行作为预测输入
        last_row = df_features.iloc[[-1]]
        feature_cols = self.feature_eng.get_feature_names()
        feature_cols = [col for col in feature_cols if col in last_row.columns]
        
        X_pred = last_row[feature_cols]
        
        if self.is_trained and self.model is not None:
            # 使用训练好的模型
            prob = self.model.predict(X_pred)[0]
            prediction = 1 if prob > 0.5 else 0
            confidence = abs(prob - 0.5) * 2  # 归一化到0-1
            
            # 计算涨跌幅预测
            expected_return = self._estimate_return(df, prob)
            
            return {
                'prediction': prediction,  # 1=上涨, 0=下跌
                'probability': prob,
                'confidence': confidence,
                'expected_return': expected_return,  # 预期涨跌幅
                'horizon_days': self.prediction_horizon,
                'model_type': 'lightgbm'
            }
        else:
            # 规则-based预测（fallback）
            return self._rule_based_prediction(last_row, df)
    
    def _estimate_return(self, df: pd.DataFrame, prob: float) -> float:
        """
        估算预期涨跌幅
        
        基于历史波动率和预测概率计算预期收益
        
        Args:
            df: 历史数据
            prob: 上涨概率
            
        Returns:
            预期涨跌幅（百分比）
        """
        try:
            # 计算历史波动率
            returns = df['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(self.prediction_horizon)  # 预测期波动率
            
            # 基于概率和波动率计算预期收益
            # 上涨时：prob * volatility * 0.5
            # 下跌时：(1-prob) * volatility * -0.5
            if prob > 0.5:
                expected_return = (prob - 0.5) * 2 * volatility * 100  # 转为百分比
            else:
                expected_return = (prob - 0.5) * 2 * volatility * 100
            
            # 限制在合理范围内
            expected_return = max(-20, min(20, expected_return))
            
            return round(expected_return, 2)
        except Exception as e:
            logger.error(f"估算涨跌幅失败: {e}")
            return 0.0
    
    def _rule_based_prediction(self, row: pd.DataFrame, df: pd.DataFrame = None) -> Dict:
        """
        基于规则的预测（当模型不可用时）
        """
        score = 0
        signals = []
        
        # RSI信号
        if 'rsi_12' in row.columns:
            rsi = row['rsi_12'].iloc[0]
            if rsi < 30:
                score += 2
                signals.append('RSI超卖')
            elif rsi > 70:
                score -= 2
                signals.append('RSI超买')
        
        # MACD信号
        if 'macd_hist' in row.columns:
            macd_hist = row['macd_hist'].iloc[0]
            if macd_hist > 0:
                score += 1
                signals.append('MACD金叉')
            else:
                score -= 1
                signals.append('MACD死叉')
        
        # 均线信号
        if 'ma5_ma10_cross' in row.columns:
            if row['ma5_ma10_cross'].iloc[0] == 1:
                score += 1
                signals.append('短期均线向上')
            else:
                score -= 1
                signals.append('短期均线向下')
        
        # 布林带位置
        if 'bb_position' in row.columns:
            bb_pos = row['bb_position'].iloc[0]
            if bb_pos < 0.2:
                score += 1
                signals.append('接近布林带下轨')
            elif bb_pos > 0.8:
                score -= 1
                signals.append('接近布林带上轨')
        
        # 成交量信号
        if 'volume_ratio_5' in row.columns:
            vol_ratio = row['volume_ratio_5'].iloc[0]
            if vol_ratio > 1.5:
                score += 1
                signals.append('放量')
            elif vol_ratio < 0.5:
                score -= 1
                signals.append('缩量')
        
        # 转换为概率
        prob = 0.5 + score * 0.1
        prob = max(0.1, min(0.9, prob))  # 限制在0.1-0.9
        
        prediction = 1 if prob > 0.5 else 0
        confidence = abs(prob - 0.5) * 2
        
        # 估算涨跌幅
        expected_return = 0.0
        if df is not None:
            expected_return = self._estimate_return(df, prob)
        
        return {
            'prediction': prediction,
            'probability': prob,
            'confidence': confidence,
            'expected_return': expected_return,
            'horizon_days': self.prediction_horizon,
            'model_type': 'rule_based',
            'signals': signals
        }
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """获取特征重要性"""
        if not self.is_trained or self.model is None:
            return None
        
        importance = self.model.feature_importance(importance_type='gain')
        feature_names = self.model.feature_name()
        
        df_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return df_importance
