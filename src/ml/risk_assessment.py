"""
风险评估模块
计算波动率、最大回撤、风险等级等指标
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class RiskAssessor:
    """风险评估器"""
    
    def __init__(self, risk_free_rate: float = 0.03):
        """
        Args:
            risk_free_rate: 无风险利率（年化，默认3%）
        """
        self.risk_free_rate = risk_free_rate
    
    def assess(self, df: pd.DataFrame) -> Dict:
        """
        综合风险评估
        
        Args:
            df: 包含OHLCV的历史数据
        
        Returns:
            风险指标字典
        """
        if df.empty or len(df) < 20:
            return {
                'risk_level': 'UNKNOWN',
                'error': '数据不足'
            }
        
        metrics = {}
        
        # 1. 波动率指标
        metrics.update(self._calculate_volatility(df))
        
        # 2. 回撤指标
        metrics.update(self._calculate_drawdown(df))
        
        # 3. 风险收益指标
        metrics.update(self._calculate_risk_return(df))
        
        # 4. 流动性风险
        metrics.update(self._calculate_liquidity_risk(df))
        
        # 5. 综合风险等级
        metrics['risk_level'] = self._determine_risk_level(metrics)
        
        # 6. 风险评分（0-100，越高越危险）
        metrics['risk_score'] = self._calculate_risk_score(metrics)
        
        return metrics
    
    def _calculate_volatility(self, df: pd.DataFrame) -> Dict:
        """波动率指标"""
        returns = df['close'].pct_change().dropna()
        
        if len(returns) < 5:
            return {
                'volatility_5d': None,
                'volatility_20d': None,
                'volatility_60d': None,
                'annualized_volatility': None
            }
        
        # 短期波动率
        vol_5d = returns.tail(5).std() * np.sqrt(252) if len(returns) >= 5 else None
        vol_20d = returns.tail(20).std() * np.sqrt(252) if len(returns) >= 20 else None
        vol_60d = returns.tail(60).std() * np.sqrt(252) if len(returns) >= 60 else None
        
        # 年化波动率
        annual_vol = returns.std() * np.sqrt(252)
        
        return {
            'volatility_5d': vol_5d,
            'volatility_20d': vol_20d,
            'volatility_60d': vol_60d,
            'annualized_volatility': annual_vol
        }
    
    def _calculate_drawdown(self, df: pd.DataFrame) -> Dict:
        """回撤指标"""
        if len(df) < 5:
            return {
                'max_drawdown': None,
                'current_drawdown': None,
                'avg_drawdown': None
            }
        
        # 计算累计收益
        cum_returns = (1 + df['close'].pct_change()).cumprod()
        
        # 计算历史最高值
        running_max = cum_returns.cummax()
        
        # 计算回撤
        drawdown = (cum_returns - running_max) / running_max
        
        max_drawdown = drawdown.min()
        current_drawdown = drawdown.iloc[-1]
        
        # 平均回撤（只计算负值）
        negative_drawdowns = drawdown[drawdown < 0]
        avg_drawdown = negative_drawdowns.mean() if len(negative_drawdowns) > 0 else 0
        
        return {
            'max_drawdown': max_drawdown,
            'current_drawdown': current_drawdown,
            'avg_drawdown': avg_drawdown
        }
    
    def _calculate_risk_return(self, df: pd.DataFrame) -> Dict:
        """风险收益指标"""
        if len(df) < 20:
            return {
                'sharpe_ratio': None,
                'sortino_ratio': None,
                'calmar_ratio': None,
                'return_20d': None,
                'return_60d': None
            }
        
        returns = df['close'].pct_change().dropna()
        
        # 收益率
        return_20d = (df['close'].iloc[-1] / df['close'].iloc[-20] - 1) if len(df) >= 20 else None
        return_60d = (df['close'].iloc[-1] / df['close'].iloc[-60] - 1) if len(df) >= 60 else None
        
        # 年化收益率
        if return_60d is not None:
            annual_return = (1 + return_60d) ** (252 / 60) - 1
        elif return_20d is not None:
            annual_return = (1 + return_20d) ** (252 / 20) - 1
        else:
            annual_return = 0
        
        # 年化波动率
        annual_vol = returns.std() * np.sqrt(252)
        
        # 夏普比率
        sharpe_ratio = (annual_return - self.risk_free_rate) / (annual_vol + 1e-10) if annual_vol > 0 else 0
        
        # 索提诺比率（只考虑下行风险）
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 1e-10
        sortino_ratio = (annual_return - self.risk_free_rate) / downside_std
        
        # 卡玛比率（收益/最大回撤）
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 1e-10
        calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'return_20d': return_20d,
            'return_60d': return_60d,
            'annualized_return': annual_return
        }
    
    def _calculate_liquidity_risk(self, df: pd.DataFrame) -> Dict:
        """流动性风险"""
        if len(df) < 5:
            return {
                'avg_volume_5d': None,
                'volume_std_5d': None,
                'amihud_illiquidity': None
            }
        
        # 平均成交量
        avg_volume_5d = df['volume'].tail(5).mean()
        volume_std_5d = df['volume'].tail(5).std()
        
        # Amihud非流动性指标（价格变化/成交量）
        returns = df['close'].pct_change().abs()
        volume_in_yuan = df['volume'] * df['close']  # 假设volume是股数
        
        amihud = (returns / (volume_in_yuan + 1e-10)).tail(20).mean() if len(df) >= 20 else None
        
        return {
            'avg_volume_5d': avg_volume_5d,
            'volume_std_5d': volume_std_5d,
            'amihud_illiquidity': amihud
        }
    
    def _determine_risk_level(self, metrics: Dict) -> str:
        """
        确定风险等级
        
        Returns:
            LOW, MEDIUM, HIGH, VERY_HIGH
        """
        risk_factors = []
        
        # 波动率风险
        if metrics.get('annualized_volatility') is not None:
            vol = metrics['annualized_volatility']
            if vol > 0.5:
                risk_factors.append(3)
            elif vol > 0.3:
                risk_factors.append(2)
            elif vol > 0.2:
                risk_factors.append(1)
        
        # 回撤风险
        if metrics.get('max_drawdown') is not None:
            dd = abs(metrics['max_drawdown'])
            if dd > 0.3:
                risk_factors.append(3)
            elif dd > 0.2:
                risk_factors.append(2)
            elif dd > 0.1:
                risk_factors.append(1)
        
        # 夏普比率
        if metrics.get('sharpe_ratio') is not None:
            sharpe = metrics['sharpe_ratio']
            if sharpe < 0:
                risk_factors.append(2)
            elif sharpe < 0.5:
                risk_factors.append(1)
        
        # 综合判断
        if not risk_factors:
            return 'UNKNOWN'
        
        avg_risk = np.mean(risk_factors)
        
        if avg_risk >= 2.5:
            return 'VERY_HIGH'
        elif avg_risk >= 1.5:
            return 'HIGH'
        elif avg_risk >= 0.8:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _calculate_risk_score(self, metrics: Dict) -> float:
        """
        计算风险评分（0-100）
        
        分数越高，风险越大
        """
        score = 50  # 基础分
        
        # 波动率贡献（0-30分）
        if metrics.get('annualized_volatility') is not None:
            vol = metrics['annualized_volatility']
            vol_score = min(30, vol * 60)  # 0.5波动率=30分
            score += vol_score - 15  # 中心化
        
        # 回撤贡献（0-25分）
        if metrics.get('max_drawdown') is not None:
            dd = abs(metrics['max_drawdown'])
            dd_score = min(25, dd * 50)  # 0.5回撤=25分
            score += dd_score - 12.5
        
        # 夏普比率贡献（-20到+20分）
        if metrics.get('sharpe_ratio') is not None:
            sharpe = metrics['sharpe_ratio']
            sharpe_score = max(-20, min(20, -sharpe * 20))  # 负夏普加分
            score += sharpe_score
        
        # 限制在0-100
        score = max(0, min(100, score))
        
        return round(score, 1)
    
    def format_risk_report(self, metrics: Dict) -> str:
        """
        格式化风险报告
        """
        lines = []
        lines.append("## 📊 风险评估报告\n")
        
        # 风险等级
        risk_level = metrics.get('risk_level', 'UNKNOWN')
        risk_score = metrics.get('risk_score', 50)
        
        emoji_map = {
            'LOW': '🟢',
            'MEDIUM': '🟡',
            'HIGH': '🟠',
            'VERY_HIGH': '🔴',
            'UNKNOWN': '⚪'
        }
        emoji = emoji_map.get(risk_level, '⚪')
        
        lines.append(f"**风险等级**: {emoji} {risk_level}")
        lines.append(f"**风险评分**: {risk_score}/100\n")
        
        # 波动率
        lines.append("### 波动率指标")
        if metrics.get('volatility_5d') is not None:
            lines.append(f"- 5日波动率: {metrics['volatility_5d']:.2%}")
        if metrics.get('volatility_20d') is not None:
            lines.append(f"- 20日波动率: {metrics['volatility_20d']:.2%}")
        if metrics.get('annualized_volatility') is not None:
            lines.append(f"- 年化波动率: {metrics['annualized_volatility']:.2%}")
        
        # 回撤
        lines.append("\n### 回撤指标")
        if metrics.get('max_drawdown') is not None:
            lines.append(f"- 最大回撤: {metrics['max_drawdown']:.2%}")
        if metrics.get('current_drawdown') is not None:
            lines.append(f"- 当前回撤: {metrics['current_drawdown']:.2%}")
        
        # 风险收益
        lines.append("\n### 风险收益指标")
        if metrics.get('sharpe_ratio') is not None:
            lines.append(f"- 夏普比率: {metrics['sharpe_ratio']:.2f}")
        if metrics.get('sortino_ratio') is not None:
            lines.append(f"- 索提诺比率: {metrics['sortino_ratio']:.2f}")
        if metrics.get('calmar_ratio') is not None:
            lines.append(f"- 卡玛比率: {metrics['calmar_ratio']:.2f}")
        
        # 收益率
        lines.append("\n### 收益率")
        if metrics.get('return_20d') is not None:
            lines.append(f"- 20日收益: {metrics['return_20d']:.2%}")
        if metrics.get('return_60d') is not None:
            lines.append(f"- 60日收益: {metrics['return_60d']:.2%}")
        if metrics.get('annualized_return') is not None:
            lines.append(f"- 年化收益: {metrics['annualized_return']:.2%}")
        
        return '\n'.join(lines)
