"""
飞书应用机器人 - 可交互版本
支持在飞书中发送消息，机器人回复预测结果
"""

import os
import json
import hashlib
import time
import requests
from flask import Flask, request, jsonify
from typing import Dict, Optional
import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from stock_prediction import StockPredictionSystem
from config import MAX_CONCURRENT_PREDICTIONS

app = Flask(__name__)

# 飞书应用配置
APP_ID = os.getenv('FEISHU_APP_ID', '')
APP_SECRET = os.getenv('FEISHU_APP_SECRET', '')
VERIFICATION_TOKEN = os.getenv('FEISHU_VERIFICATION_TOKEN', '')
ENCRYPT_KEY = os.getenv('FEISHU_ENCRYPT_KEY', '')

# 初始化预测系统
prediction_system = StockPredictionSystem()


class FeishuAppBot:
    """飞书应用机器人"""
    
    def __init__(self, app_id: str, app_secret: str):
        self.app_id = app_id
        self.app_secret = app_secret
        self.tenant_access_token = None
        self.token_expires_at = 0
    
    def get_tenant_access_token(self) -> str:
        """获取 tenant_access_token"""
        # 如果token未过期，直接返回
        if self.tenant_access_token and time.time() < self.token_expires_at:
            return self.tenant_access_token
        
        # 获取新token
        url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
        payload = {
            "app_id": self.app_id,
            "app_secret": self.app_secret
        }
        
        response = requests.post(url, json=payload, timeout=10)
        data = response.json()
        
        if data.get('code') == 0:
            self.tenant_access_token = data['tenant_access_token']
            # token有效期2小时，提前5分钟刷新
            self.token_expires_at = time.time() + data['expire'] - 300
            return self.tenant_access_token
        else:
            raise Exception(f"获取token失败: {data}")
    
    def send_message(self, chat_id: str, text: str, msg_type: str = "text"):
        """发送消息到飞书"""
        token = self.get_tenant_access_token()
        
        url = "https://open.feishu.cn/open-apis/im/v1/messages"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "receive_id": chat_id,
            "msg_type": msg_type,
            "content": json.dumps({"text": text}) if msg_type == "text" else text
        }
        
        params = {
            "receive_id_type": "chat_id"
        }
        
        response = requests.post(url, headers=headers, json=payload, params=params, timeout=10)
        data = response.json()
        
        if data.get('code') == 0:
            print(f"消息发送成功: {chat_id}")
            return True
        else:
            print(f"消息发送失败: {data}")
            return False
    
    def reply_message(self, message_id: str, text: str):
        """回复消息"""
        token = self.get_tenant_access_token()
        
        url = f"https://open.feishu.cn/open-apis/im/v1/messages/{message_id}/reply"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "content": json.dumps({"text": text}),
            "msg_type": "text"
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        data = response.json()
        
        if data.get('code') == 0:
            print(f"消息回复成功: {message_id}")
            return True
        else:
            print(f"消息回复失败: {data}")
            return False


# 初始化机器人
bot = FeishuAppBot(APP_ID, APP_SECRET)


def handle_message(event: Dict) -> Dict:
    """处理消息事件"""
    try:
        message = event.get('message', {})
        chat_id = message.get('chat_id', '')
        message_id = message.get('message_id', '')
        content = message.get('content', '{}')
        
        # 解析消息内容
        content_json = json.loads(content)
        text = content_json.get('text', '').strip()
        
        print(f"收到消息: {text}")
        
        # 处理命令
        response_text = process_command(text)
        
        # 回复消息
        bot.reply_message(message_id, response_text)
        
        return {"code": 0}
    
    except Exception as e:
        print(f"处理消息失败: {e}")
        import traceback
        traceback.print_exc()
        return {"code": 1, "msg": str(e)}


def process_command(text: str) -> str:
    """处理用户命令"""
    text = text.strip()
    
    # 去掉@机器人的部分
    if text.startswith("@_user_1"):
        text = text.replace("@_user_1", "").strip()
    
    # 解析命令
    if text.startswith("预测") or text.startswith("分析"):
        return handle_predict(text)
    elif text.startswith("训练"):
        return handle_train(text)
    elif text.startswith("批量"):
        return handle_batch_predict(text)
    elif text in ["帮助", "help", "?", ""]:
        return show_help()
    else:
        return show_help()


def handle_predict(message: str) -> str:
    """处理预测命令"""
    try:
        parts = message.split()
        if len(parts) < 2:
            return "❌ 格式错误\n\n正确格式: 预测 000001 [天数]\n示例: 预测 000001 5天"
        
        stock_code = parts[1]
        horizon = 5
        
        # 解析天数
        if len(parts) >= 3:
            horizon_str = parts[2]
            if "天" in horizon_str:
                horizon = int(horizon_str.replace("天", ""))
            else:
                horizon = int(horizon_str)
        
        # 执行预测
        result = prediction_system.predict_stock(stock_code, horizon)
        report = prediction_system.format_prediction_report(result)
        
        return report
    
    except Exception as e:
        return f"❌ 预测失败: {str(e)}"


def handle_train(message: str) -> str:
    """处理训练命令"""
    try:
        parts = message.split()
        if len(parts) < 2:
            return "❌ 格式错误\n\n正确格式: 训练 000001"
        
        stock_code = parts[1]
        
        # 训练模型
        metrics = prediction_system.train_model(stock_code, 5)
        
        # 格式化结果
        class_metrics = list(metrics['classification'].values())[0]
        reg_metrics = list(metrics['regression'].values())[0]
        
        class_acc = class_metrics.get('accuracy', 0)
        reg_mse = reg_metrics.get('mse', 0)
        
        return f"""✅ 模型训练完成

股票: {stock_code}
分类准确率: {class_acc:.2%}
回归MSE: {reg_mse:.4f}

现在可以使用"预测 {stock_code}"进行预测"""
    
    except Exception as e:
        return f"❌ 训练失败: {str(e)}"


def handle_batch_predict(message: str) -> str:
    """处理批量预测命令 - 支持6个并行"""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    try:
        parts = message.split()
        if len(parts) < 2:
            return "❌ 格式错误\n\n正确格式: 批量预测 000001 600519 002594"
        
        stock_codes = parts[1:]
        
        # 定义单个股票预测函数
        def predict_single_stock(stock_code):
            try:
                result = prediction_system.predict_stock(stock_code, 5)
                direction = "📈 看涨" if result['direction']['prediction'] == 1 else "📉 跌"
                prob = result['direction']['probability']
                ret = result['expected_return']['prediction'] * 100
                risk = result['risk_level']
                
                return f"{stock_code}: {direction} (概率{prob:.0%}, 预期{ret:+.1f}%, 风险{risk})"
            except Exception as e:
                return f"{stock_code}: ❌ {str(e)}"
        
        # 并行预测，最多6个同时
        results = []
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_PREDICTIONS) as executor:
            futures = [executor.submit(predict_single_stock, code) for code in stock_codes]
            for future in as_completed(futures):
                results.append(future.result())
        
        report = f"📊 批量预测结果（{MAX_CONCURRENT_PREDICTIONS}路并行）\n\n" + "\n".join(results)
        return report
    
    except Exception as e:
        return f"❌ 批量预测失败: {str(e)}"


def show_help() -> str:
    """显示帮助信息"""
    return """🤖 股票预测机器人使用指南

📌 支持的命令:

1️⃣ 预测单只股票
   预测 000001
   预测 000001 10天
   
2️⃣ 训练模型
   训练 000001
   
3️⃣ 批量预测
   批量预测 000001 600519 002594

4️⃣ 查看帮助
   帮助

💡 示例:
- "预测 000001" - 预测平安银行未来5天走势
- "预测 600519 10天" - 预测贵州茅台未来10天走势
- "训练 002594" - 训练比亚迪的预测模型
- "批量预测 000001 600519" - 同时预测多只股票

⚠️ 注意:
- 首次预测某只股票会自动训练模型
- 模型训练需要1-2分钟
- 预测结果仅供参考，不构成投资建议"""


@app.route('/webhook/event', methods=['POST'])
def webhook_event():
    """接收飞书事件回调"""
    data = request.json
    
    # 验证token
    if VERIFICATION_TOKEN and data.get('token') != VERIFICATION_TOKEN:
        return jsonify({"code": 1, "msg": "token验证失败"}), 403
    
    # 处理URL验证（首次配置时）
    if data.get('type') == 'url_verification':
        challenge = data.get('challenge', '')
        return jsonify({"challenge": challenge})
    
    # 处理事件
    if data.get('schema') == '2.0':
        event = data.get('event', {})
        
        # 异步处理事件
        import threading
        threading.Thread(target=handle_message, args=(event,)).start()
        
        # 立即返回200，避免飞书重试
        return jsonify({"code": 0})
    
    return jsonify({"code": 0})


@app.route('/health', methods=['GET'])
def health():
    """健康检查"""
    return jsonify({
        "status": "ok",
        "timestamp": time.time()
    })


if __name__ == '__main__':
    print("=" * 60)
    print("飞书应用机器人启动")
    print("=" * 60)
    print(f"APP_ID: {APP_ID[:10]}..." if APP_ID else "APP_ID: 未配置")
    print(f"Webhook URL: /webhook/event")
    print("=" * 60)
    
    if not APP_ID or not APP_SECRET:
        print("\n⚠️ 警告: 飞书应用配置未设置")
        print("请设置环境变量:")
        print("  export FEISHU_APP_ID='your_app_id'")
        print("  export FEISHU_APP_SECRET='your_app_secret'")
        print("  export FEISHU_VERIFICATION_TOKEN='your_token'")
    
    app.run(host='0.0.0.0', port=8080, debug=False)
