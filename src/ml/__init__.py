"""
ML诊断模块
- 趋势预测：基于技术指标预测涨跌概率
- 风险评估：波动率、最大回撤、风险等级
- 集成分析：将ML诊断整合到LLM分析流程
"""

from .feature_engineering import FeatureEngineering
from .model import TrendPredictor
from .risk_assessment import RiskAssessor
from .diagnostic import MLDiagnostician, diagnose_stock
from .integration import MLAnalysisIntegrator, integrate_ml_analysis

__all__ = [
    'FeatureEngineering', 
    'TrendPredictor', 
    'RiskAssessor',
    'MLDiagnostician',
    'diagnose_stock',
    'MLAnalysisIntegrator',
    'integrate_ml_analysis'
]
