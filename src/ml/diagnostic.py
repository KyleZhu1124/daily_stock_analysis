"""
ML诊断集成模块
将ML诊断功能集成到现有分析流程中
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging
from datetime import datetime, timedelta

from .feature_engineering import FeatureEngineering
from .model import TrendPredictor
from .risk_assessment import RiskAssessor

logger = logging.getLogger(__name__)


class MLDiagnostician:
    """ML诊断器 - 整合趋势预测和风险评估"""
    
    def __init__(self, prediction_horizon: int = 5):
        """
        Args:
            prediction_horizon: 预测未来天数
        """
        self.prediction_horizon = prediction_horizon
        self.trend_predictor = TrendPredictor(prediction_horizon)
        self.risk_assessor = RiskAssessor()
        self.feature_eng = FeatureEngineering()
    
    def diagnose(self, stock_code: str, df: pd.DataFrame, train_model: bool = False) -> Dict:
        """
        对单只股票进行ML诊断
        
        Args:
            stock_code: 股票代码
            df: 历史OHLCV数据
            train_model: 是否训练模型（首次使用或定期更新时设为True）
        
        Returns:
            诊断结果字典
        """
        if df.empty or len(df) < 60:
            return {
                'stock_code': stock_code,
                'error': '数据不足，无法进行ML诊断',
                'timestamp': datetime.now().isoformat()
            }
        
        result = {
            'stock_code': stock_code,
            'timestamp': datetime.now().isoformat(),
            'data_points': len(df)
        }
        
        try:
            # 1. 训练模型（如果需要）
            if train_model:
                logger.info(f"[{stock_code}] 训练趋势预测模型...")
                train_metrics = self.trend_predictor.train(df)
                result['train_metrics'] = train_metrics
            
            # 2. 趋势预测
            logger.info(f"[{stock_code}] 进行趋势预测...")
            trend_prediction = self.trend_predictor.predict(df)
            result['trend'] = trend_prediction
            
            # 3. 风险评估
            logger.info(f"[{stock_code}] 进行风险评估...")
            risk_metrics = self.risk_assessor.assess(df)
            result['risk'] = risk_metrics
            
            # 4. 综合诊断结论
            result['conclusion'] = self._generate_conclusion(trend_prediction, risk_metrics)
            
            # 5. 生成报告
            result['report'] = self._generate_report(result)
            
        except Exception as e:
            logger.error(f"[{stock_code}] ML诊断失败: {str(e)}")
            result['error'] = str(e)
        
        return result
    
    def _generate_conclusion(self, trend: Dict, risk: Dict) -> str:
        """生成综合诊断结论"""
        conclusions = []
        
        # 趋势判断
        if 'error' in trend:
            conclusions.append("⚠️ 趋势预测不可用")
        else:
            pred = trend.get('prediction')
            prob = trend.get('probability', 0.5)
            conf = trend.get('confidence', 0)
            
            if pred == 1:
                if conf > 0.6:
                    conclusions.append(f"📈 强烈看涨（{prob:.0%}概率，{conf:.0%}置信度）")
                elif conf > 0.3:
                    conclusions.append(f"📈 偏多（{prob:.0%}概率）")
                else:
                    conclusions.append(f"📈 轻微看涨（{prob:.0%}概率）")
            else:
                if conf > 0.6:
                    conclusions.append(f"📉 强烈看跌（{1-prob:.0%}概率，{conf:.0%}置信度）")
                elif conf > 0.3:
                    conclusions.append(f"📉 偏空（{1-prob:.0%}概率）")
                else:
                    conclusions.append(f"📉 轻微看跌（{1-prob:.0%}概率）")
            
            # 添加信号
            if 'signals' in trend:
                conclusions.append(f"技术信号: {', '.join(trend['signals'])}")
        
        # 风险判断
        risk_level = risk.get('risk_level', 'UNKNOWN')
        risk_score = risk.get('risk_score', 50)
        
        risk_desc = {
            'LOW': '低风险',
            'MEDIUM': '中等风险',
            'HIGH': '高风险',
            'VERY_HIGH': '极高风险',
            'UNKNOWN': '风险未知'
        }
        
        conclusions.append(f"{risk_desc.get(risk_level, '风险未知')}（评分{risk_score}/100）")
        
        # 综合建议
        if 'error' not in trend and risk_level in ['LOW', 'MEDIUM']:
            if trend.get('prediction') == 1 and trend.get('confidence', 0) > 0.3:
                conclusions.append("💡 建议：可考虑逢低布局")
            elif trend.get('prediction') == 0 and trend.get('confidence', 0) > 0.3:
                conclusions.append("💡 建议：谨慎观望，等待企稳")
        elif risk_level in ['HIGH', 'VERY_HIGH']:
            conclusions.append("💡 建议：风险较高，控制仓位")
        
        return '\n'.join(conclusions)
    
    def _generate_report(self, result: Dict) -> str:
        """生成完整的诊断报告"""
        lines = []
        
        stock_code = result.get('stock_code', 'Unknown')
        lines.append(f"# 🔬 ML诊断报告 - {stock_code}\n")
        lines.append(f"**诊断时间**: {result.get('timestamp', 'N/A')}\n")
        
        # 训练指标（如果有）
        if 'train_metrics' in result:
            metrics = result['train_metrics']
            if 'error' not in metrics:
                lines.append("## 📚 模型训练指标")
                lines.append(f"- 交叉验证AUC: {metrics.get('cv_auc_mean', 0):.3f} ± {metrics.get('cv_auc_std', 0):.3f}")
                lines.append(f"- 训练样本数: {metrics.get('n_samples', 0)}")
                lines.append(f"- 正样本比例: {metrics.get('n_positive', 0) / metrics.get('n_samples', 1):.1%}\n")
        
        # 趋势预测
        if 'trend' in result:
            trend = result['trend']
            if 'error' not in trend:
                lines.append("## 📈 趋势预测")
                pred_text = "上涨" if trend.get('prediction') == 1 else "下跌"
                lines.append(f"- 预测方向: {pred_text}")
                lines.append(f"- 上涨概率: {trend.get('probability', 0):.1%}")
                lines.append(f"- 置信度: {trend.get('confidence', 0):.1%}")
                expected_return = trend.get('expected_return', 0)
                if expected_return > 0:
                    lines.append(f"- 预期涨幅: +{expected_return:.2f}%")
                elif expected_return < 0:
                    lines.append(f"- 预期跌幅: {expected_return:.2f}%")
                else:
                    lines.append(f"- 预期涨跌幅: {expected_return:.2f}%")
                lines.append(f"- 预测周期: {trend.get('horizon_days', 5)}天")
                lines.append(f"- 模型类型: {trend.get('model_type', 'unknown')}")
                
                if 'signals' in trend:
                    lines.append(f"- 技术信号: {', '.join(trend['signals'])}")
                lines.append("")
        
        # 风险评估
        if 'risk' in result:
            lines.append(self.risk_assessor.format_risk_report(result['risk']))
            lines.append("")
        
        # 综合结论
        if 'conclusion' in result:
            lines.append("## 💡 综合诊断结论")
            lines.append(result['conclusion'])
        
        return '\n'.join(lines)


def diagnose_stock(stock_code: str, df: pd.DataFrame, train_model: bool = False) -> Dict:
    """
    便捷函数：对单只股票进行ML诊断
    
    Args:
        stock_code: 股票代码
        df: 历史OHLCV数据
        train_model: 是否训练模型
    
    Returns:
        诊断结果字典
    """
    diagnostician = MLDiagnostician()
    return diagnostician.diagnose(stock_code, df, train_model)


if __name__ == '__main__':
    # 测试代码
    import akshare as ak
    
    # 获取测试数据
    print("获取测试数据...")
    df = ak.stock_zh_a_hist(symbol="000001", period="daily", start_date="20240101", adjust="qfq")
    
    # 进行诊断
    print("\n进行ML诊断...")
    result = diagnose_stock("000001", df, train_model=True)
    
    # 输出报告
    print("\n" + "="*60)
    print(result.get('report', 'No report generated'))
