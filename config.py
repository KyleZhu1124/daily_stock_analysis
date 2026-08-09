"""
系统配置文件
统一管理所有并发和性能参数
"""

# 并发配置
MAX_CONCURRENT_PREDICTIONS = 6  # 最大并行预测数
MAX_CONCURRENT_TRAINING = 6     # 最大并行训练数
MAX_CONCURRENT_MODELS = 6       # 最大并行模型数

# GPU配置
GPU_DEVICE = 0  # 使用第一张GPU

# 模型配置
N_ESTIMATORS = 500
MAX_DEPTH = 8
LEARNING_RATE = 0.05

# 数据配置
DEFAULT_PREDICTION_HORIZON = 5  # 默认预测天数
CACHE_EXPIRY_DAYS = 1           # 数据缓存过期天数
