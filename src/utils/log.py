import logging
import pytz
from datetime import datetime

# 设置时区为中国时区
china_tz = pytz.timezone('Asia/Shanghai')

# 自定义日志格式化器，使用中国时区
class ChinaTimeFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        ct = datetime.fromtimestamp(record.created, china_tz)
        if datefmt:
            s = ct.strftime(datefmt)
        else:
            s = ct.strftime("%Y-%m-%d %H:%M:%S")
        return s

def setup_logger(log_file, log_level=logging.INFO):
    """
    设置日志记录器
    
    Args:
        log_file (str): 日志文件路径
        log_level (int): 日志级别，默认为 INFO
    
    Returns:
        logging.Logger: 配置好的日志记录器
    """
    # 配置基本日志设置
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename=log_file,
        filemode="a",
    )
    
    # 获取根日志记录器并设置自定义格式化器
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        formatter = ChinaTimeFormatter(
            fmt="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
    
    return root_logger
