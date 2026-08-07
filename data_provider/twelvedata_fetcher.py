#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Twelve Data 数据获取器
支持全球股票数据，免费 800 次/天
"""

import os
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import requests

from data_provider.base import BaseFetcher
from data_provider.realtime_types import UnifiedRealtimeQuote

logger = logging.getLogger(__name__)


class TwelveDataFetcher(BaseFetcher):
    """Twelve Data 数据获取器"""
    
    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("TWELVEDATA_API_KEY")
        self.base_url = "https://api.twelvedata.com"
        self.enabled = bool(self.api_key)
        
        if self.enabled:
            logger.info("[数据源] Twelve Data 已启用")
        else:
            logger.debug("[数据源] Twelve Data 未配置 API Key")
    
    def is_available(self) -> bool:
        """检查数据源是否可用"""
        return self.enabled
    
    def get_realtime_quote(self, stock_code: str) -> Optional[UnifiedRealtimeQuote]:
        """
        获取实时行情
        
        Args:
            stock_code: 股票代码（A 股格式如 600519.SH）
            
        Returns:
            UnifiedRealtimeQuote 对象
        """
        if not self.enabled:
            return None
        
        try:
            # 转换股票代码格式
            symbol = self._convert_symbol(stock_code)
            
            # 调用 Twelve Data API
            url = f"{self.base_url}/quote"
            params = {
                "symbol": symbol,
                "apikey": self.api_key,
                "outputsize": 1
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if "code" in data and data["code"] != 200:
                logger.warning(f"Twelve Data API 错误: {data.get('message', 'Unknown error')}")
                return None
            
            # 解析响应
            return self._parse_quote(data, stock_code)
            
        except Exception as e:
            logger.error(f"[TwelveData] 获取 {stock_code} 实时行情失败: {e}")
            return None
    
    def _parse_quote(self, data: Dict[str, Any], original_code: str) -> Optional[UnifiedRealtimeQuote]:
        """解析行情数据"""
        try:
            symbol = data.get("symbol", "")
            
            # 提取价格数据
            current_price = float(data.get("price", 0))
            prev_close = float(data.get("previous_close", 0))
            
            # 计算涨跌
            change_amount = current_price - prev_close if prev_close > 0 else 0
            change_pct = (change_amount / prev_close * 100) if prev_close > 0 else 0
            
            return UnifiedRealtimeQuote(
                code=original_code,
                name=data.get("name", ""),
                price=current_price,
                change_amount=change_amount,
                change_pct=change_pct,
                open_price=float(data.get("open", 0)),
                high=float(data.get("high", 0)),
                low=float(data.get("low", 0)),
                pre_close=prev_close,
                volume=int(data.get("volume", 0)),
                amount=0,  # Twelve Data 不提供成交额
                update_time=data.get("timestamp", datetime.now().isoformat()),
                source="twelvedata"
            )
        except Exception as e:
            logger.error(f"[TwelveData] 解析行情数据失败: {e}")
            return None
    
    def _convert_symbol(self, stock_code: str) -> str:
        """
        转换股票代码格式
        
        A 股: 600519.SH -> 600519.SHH (Twelve Data 格式)
        港股: 00700.HK -> 00700.HK
        美股: AAPL -> AAPL
        """
        if stock_code.endswith(".SH"):
            # 上海证券交易所
            return stock_code.replace(".SH", ".SHH")
        elif stock_code.endswith(".SZ"):
            # 深圳证券交易所
            return stock_code.replace(".SZ", ".SHZ")
        elif stock_code.endswith(".HK"):
            # 港股
            return stock_code
        else:
            # 美股或其他
            return stock_code
    
    def get_daily_data(self, stock_code: str, days: int = 30) -> Optional[List[Dict[str, Any]]]:
        """
        获取日线数据
        
        Args:
            stock_code: 股票代码
            days: 获取天数
            
        Returns:
            日线数据列表
        """
        if not self.enabled:
            return None
        
        try:
            symbol = self._convert_symbol(stock_code)
            
            url = f"{self.base_url}/time_series"
            params = {
                "symbol": symbol,
                "interval": "1day",
                "outputsize": days,
                "apikey": self.api_key
            }
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            if "code" in data and data["code"] != 200:
                logger.warning(f"Twelve Data API 错误: {data.get('message', 'Unknown error')}")
                return None
            
            values = data.get("values", [])
            result = []
            
            for item in values:
                result.append({
                    "date": item.get("datetime"),
                    "open": float(item.get("open", 0)),
                    "high": float(item.get("high", 0)),
                    "low": float(item.get("low", 0)),
                    "close": float(item.get("close", 0)),
                    "volume": int(item.get("volume", 0))
                })
            
            return result
            
        except Exception as e:
            logger.error(f"[TwelveData] 获取 {stock_code} 日线数据失败: {e}")
            return None
