"""
集成模型架构
结合LightGBM、XGBoost、CatBoost和神经网络
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')


class EnsembleModel:
    """集成模型类"""
    
    def __init__(self, task_type: str = 'classification'):
        """
        Args:
            task_type: 'classification' (涨跌预测) 或 'regression' (涨幅预测)
        """
        self.task_type = task_type
        self.models = {}
        self.weights = {}
        self.is_trained = False
        
    def _init_models(self):
        """初始化各个模型"""
        if self.task_type == 'classification':
            # 分类任务：预测涨跌
            self.models['lightgbm'] = lgb.LGBMClassifier(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.05,
                num_leaves=63,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            )
            
            self.models['xgboost'] = xgb.XGBClassifier(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0
            )
            
            self.models['catboost'] = CatBoostRegressor(
                iterations=500,
                depth=8,
                learning_rate=0.05,
                l2_leaf_reg=3,
                random_seed=42,
                verbose=False
            )
            
        else:
            # 回归任务：预测涨幅
            self.models['lightgbm'] = lgb.LGBMRegressor(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.05,
                num_leaves=63,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            )
            
            self.models['xgboost'] = xgb.XGBRegressor(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0
            )
            
            self.models['catboost'] = CatBoostRegressor(
                iterations=500,
                depth=8,
                learning_rate=0.05,
                l2_leaf_reg=3,
                random_seed=42,
                verbose=False
            )
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: pd.DataFrame = None, y_val: pd.Series = None) -> Dict:
        """训练集成模型"""
        self._init_models()
        
        metrics = {}
        
        # 训练每个模型
        for name, model in self.models.items():
            if name == 'catboost' and self.task_type == 'classification':
                # CatBoost分类需要特殊处理
                model.fit(X_train, y_train, eval_set=(X_val, y_val) if X_val is not None else None, verbose=False)
            else:
                model.fit(X_train, y_train)
            
            # 评估
            if X_val is not None:
                y_pred = model.predict(X_val)
                
                if self.task_type == 'classification':
                    if hasattr(model, 'predict_proba'):
                        y_prob = model.predict_proba(X_val)[:, 1]
                        auc = roc_auc_score(y_val, y_prob)
                        acc = accuracy_score(y_val, y_pred)
                        metrics[name] = {'auc': auc, 'accuracy': acc}
                    else:
                        mse = mean_squared_error(y_val, y_pred)
                        metrics[name] = {'mse': mse}
                else:
                    mse = mean_squared_error(y_val, y_pred)
                    metrics[name] = {'mse': mse}
        
        # 计算权重（基于验证集性能）
        if X_val is not None and self.task_type == 'classification':
            total_auc = sum(m.get('auc', 0) for m in metrics.values())
            if total_auc > 0:
                self.weights = {name: m.get('auc', 0) / total_auc 
                               for name, m in metrics.items()}
            else:
                self.weights = {name: 1.0 / len(self.models) for name in self.models}
        else:
            self.weights = {name: 1.0 / len(self.models) for name in self.models}
        
        print(f"\n模型权重: {self.weights}")
        self.is_trained = True
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> Dict:
        """集成预测"""
        if not self.is_trained:
            raise ValueError("模型未训练")
        
        predictions = {}
        
        for name, model in self.models.items():
            if self.task_type == 'classification' and hasattr(model, 'predict_proba'):
                predictions[name] = model.predict_proba(X)[:, 1]
            else:
                predictions[name] = model.predict(X)
        
        # 加权集成
        if self.task_type == 'classification':
            ensemble_pred = sum(predictions[name] * self.weights[name] 
                               for name in self.models)
            final_pred = (ensemble_pred > 0.5).astype(int)
            
            return {
                'prediction': final_pred[0],
                'probability': ensemble_pred[0],
                'individual_predictions': {name: pred[0] for name, pred in predictions.items()},
                'weights': self.weights
            }
        else:
            ensemble_pred = sum(predictions[name] * self.weights[name] 
                               for name in self.models)
            
            return {
                'prediction': ensemble_pred[0],
                'individual_predictions': {name: pred[0] for name, pred in predictions.items()},
                'weights': self.weights
            }
    
    def get_feature_importance(self) -> Dict:
        """获取特征重要性"""
        importance_dict = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance_dict[name] = model.feature_importances_
        
        return importance_dict


class StockPredictor:
    """股票预测器 - 整合特征工程和模型"""
    
    def __init__(self, prediction_horizon: int = 5):
        """
        Args:
            prediction_horizon: 预测未来天数
        """
        self.prediction_horizon = prediction_horizon
        self.classification_model = EnsembleModel(task_type='classification')
        self.regression_model = EnsembleModel(task_type='regression')
        self.is_trained = False
        
    def prepare_data(self, df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """准备训练数据"""
        # 创建标签
        # 分类标签：未来N天涨幅>2%为1，否则为0
        future_return = df['close'].shift(-self.prediction_horizon) / df['close'] - 1
        y_class = (future_return > 0.02).astype(int)
        
        # 回归标签：未来N天的实际涨幅
        y_reg = future_return
        
        # 删除NaN
        valid_mask = ~(y_class.isna() | y_reg.isna())
        
        # 检查特征列是否存在
        available_features = [col for col in feature_cols if col in df.columns]
        
        X = df[available_features][valid_mask]
        y_class = y_class[valid_mask]
        y_reg = y_reg[valid_mask]
        
        # 填充缺失值
        X = X.fillna(0)
        
        return X, y_class, y_reg
    
    def train(self, df: pd.DataFrame, feature_cols: List[str], 
              val_ratio: float = 0.2) -> Dict:
        """训练模型"""
        print("准备数据...")
        X, y_class, y_reg = self.prepare_data(df, feature_cols)
        
        # 时间序列分割
        split_idx = int(len(X) * (1 - val_ratio))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_class_train, y_class_val = y_class.iloc[:split_idx], y_class.iloc[split_idx:]
        y_reg_train, y_reg_val = y_reg.iloc[:split_idx], y_reg.iloc[split_idx:]
        
        print(f"训练集大小: {len(X_train)}, 验证集大小: {len(X_val)}")
        
        # 训练分类模型
        class_metrics = self.classification_model.train(
            X_train, y_class_train, X_val, y_class_val
        )
        
        # 训练回归模型
        reg_metrics = self.regression_model.train(
            X_train, y_reg_train, X_val, y_reg_val
        )
        
        self.is_trained = True
        
        return {
            'classification': class_metrics,
            'regression': reg_metrics
        }
    
    def predict(self, df: pd.DataFrame, feature_cols: List[str]) -> Dict:
        """预测"""
        if not self.is_trained:
            raise ValueError("模型未训练")
        
        # 准备数据（只取最后一行）
        X = df[feature_cols].iloc[[-1]].fillna(0)
        
        # 分类预测
        class_pred = self.classification_model.predict(X)
        
        # 回归预测
        reg_pred = self.regression_model.predict(X)
        
        return {
            'direction': class_pred,
            'expected_return': reg_pred,
            'confidence': abs(class_pred['probability'] - 0.5) * 2
        }
