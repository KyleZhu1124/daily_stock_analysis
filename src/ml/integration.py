"""
ML诊断集成到分析流程
将ML诊断结果合并到现有的LLM分析报告中
"""

import pandas as pd
from typing import Dict, Optional
import logging
from datetime import datetime

from .diagnostic import MLDiagnostician

logger = logging.getLogger(__name__)


class MLAnalysisIntegrator:
    """ML分析集成器"""
    
    def __init__(self, enable_ml: bool = True, train_on_first_run: bool = True):
        """
        Args:
            enable_ml: 是否启用ML诊断
            train_on_first_run: 首次运行时是否训练模型
        """
        self.enable_ml = enable_ml
        self.train_on_first_run = train_on_first_run
        self.diagnostician = MLDiagnostician() if enable_ml else None
        self.model_trained = False
    
    def integrate_ml_diagnostics(self, stock_code: str, df: pd.DataFrame, 
                                  llm_analysis: Optional[str] = None) -> Dict:
        """
        集成ML诊断到分析流程
        
        Args:
            stock_code: 股票代码
            df: 历史OHLCV数据
            llm_analysis: LLM分析结果（可选）
        
        Returns:
            集成后的分析结果
        """
        if not self.enable_ml:
            return {
                'stock_code': stock_code,
                'ml_enabled': False,
                'llm_analysis': llm_analysis
            }
        
        result = {
            'stock_code': stock_code,
            'ml_enabled': True,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # 进行ML诊断
            should_train = self.train_on_first_run and not self.model_trained
            ml_result = self.diagnostician.diagnose(stock_code, df, train_model=should_train)
            
            if should_train:
                self.model_trained = True
            
            result['ml_diagnostics'] = ml_result
            
            # 合并LLM分析和ML诊断
            if llm_analysis:
                result['combined_analysis'] = self._combine_analyses(llm_analysis, ml_result)
            else:
                result['combined_analysis'] = ml_result.get('report', '')
            
        except Exception as e:
            logger.error(f"[{stock_code}] ML集成失败: {str(e)}")
            result['error'] = str(e)
            result['combined_analysis'] = llm_analysis or f"ML诊断失败: {str(e)}"
        
        return result
    
    def _combine_analyses(self, llm_analysis: str, ml_result: Dict) -> str:
        """
        合并LLM分析和ML诊断结果
        """
        sections = []
        
        # LLM分析部分
        sections.append("## 🤖 AI智能分析")
        sections.append(llm_analysis)
        sections.append("")
        
        # ML诊断部分
        if 'report' in ml_result:
            sections.append("---")
            sections.append("")
            sections.append(ml_result['report'])
        
        # 综合建议
        sections.append("---")
        sections.append("")
        sections.append("## 🎯 综合投资建议")
        
        # 从ML结果提取关键信息
        trend = ml_result.get('trend', {})
        risk = ml_result.get('risk', {})
        
        trend_pred = trend.get('prediction')
        trend_conf = trend.get('confidence', 0)
        risk_level = risk.get('risk_level', 'UNKNOWN')
        risk_score = risk.get('risk_score', 50)
        
        # 生成建议
        if trend_pred == 1 and trend_conf > 0.5 and risk_level in ['LOW', 'MEDIUM']:
            sections.append("✅ **强烈建议买入**")
            sections.append("- 技术面看涨信号明确")
            sections.append("- 风险可控")
            sections.append("- 建议仓位: 15-20%")
        elif trend_pred == 1 and risk_level in ['LOW', 'MEDIUM']:
            sections.append("🟢 **可以考虑买入**")
            sections.append("- 趋势偏多")
            sections.append("- 风险适中")
            sections.append("- 建议仓位: 10-15%")
        elif trend_pred == 0 and risk_level in ['HIGH', 'VERY_HIGH']:
            sections.append("🔴 **建议回避**")
            sections.append("- 技术面看跌")
            sections.append("- 风险较高")
            sections.append("- 不建议介入")
        elif trend_pred == 0:
            sections.append("🟡 **谨慎观望**")
            sections.append("- 趋势偏空")
            sections.append("- 等待企稳信号")
        else:
            sections.append("⚪ **中性观望**")
            sections.append("- 信号不明确")
            sections.append("- 建议观望")
        
        # 风险提示
        sections.append("")
        sections.append("⚠️ **风险提示**: ML诊断基于历史数据，仅供参考，不构成投资建议。")
        
        return '\n'.join(sections)
    
    def format_for_notification(self, result: Dict) -> str:
        """
        格式化用于通知的消息
        """
        stock_code = result.get('stock_code', 'Unknown')
        
        if not result.get('ml_enabled'):
            return result.get('combined_analysis', 'ML诊断未启用')
        
        ml_result = result.get('ml_diagnostics', {})
        
        # 提取关键信息
        trend = ml_result.get('trend', {})
        risk = ml_result.get('risk', {})
        
        # 生成简洁版本
        lines = []
        lines.append(f"📊 **{stock_code} ML诊断摘要**\n")
        
        # 趋势
        if 'error' not in trend:
            pred = "📈 看涨" if trend.get('prediction') == 1 else "📉 看跌"
            prob = trend.get('probability', 0.5)
            conf = trend.get('confidence', 0)
            lines.append(f"**趋势**: {pred} (概率{prob:.0%}, 置信度{conf:.0%})")
        
        # 风险
        risk_level = risk.get('risk_level', 'UNKNOWN')
        risk_score = risk.get('risk_score', 50)
        emoji_map = {
            'LOW': '🟢',
            'MEDIUM': '🟡',
            'HIGH': '🟠',
            'VERY_HIGH': '🔴',
            'UNKNOWN': '⚪'
        }
        emoji = emoji_map.get(risk_level, '⚪')
        lines.append(f"**风险**: {emoji} {risk_level} ({risk_score}/100)")
        
        # 结论
        conclusion = ml_result.get('conclusion', '')
        if conclusion:
            lines.append(f"\n💡 {conclusion}")
        
        return '\n'.join(lines)


# 便捷函数
def integrate_ml_analysis(stock_code: str, df: pd.DataFrame, 
                          llm_analysis: Optional[str] = None,
                          enable_ml: bool = True) -> Dict:
    """
    便捷函数：集成ML分析
    
    Args:
        stock_code: 股票代码
        df: 历史OHLCV数据
        llm_analysis: LLM分析结果
        enable_ml: 是否启用ML
    
    Returns:
        集成后的分析结果
    """
    integrator = MLAnalysisIntegrator(enable_ml=enable_ml)
    return integrator.integrate_ml_diagnostics(stock_code, df, llm_analysis)
