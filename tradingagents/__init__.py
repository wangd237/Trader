"""Trader：基于 TradingAgents 衍生的多 Agent 市场研究框架。"""

__version__ = "0.7.0"

from dotenv import find_dotenv, load_dotenv

# 在导入包时加载 .env，使 DEFAULT_CONFIG 的环境变量覆盖规则和
# 所有 LLM 客户端无论从哪个入口启动都能读取密钥。
# usecwd=True 从当前工作目录查找配置文件，确保已安装的命令行入口
# 读取项目的 .env，而不是在 site-packages 附近寻找。
# 已经由调用者导出的变量不会被覆盖。
load_dotenv(find_dotenv(usecwd=True))
load_dotenv(find_dotenv(".env.enterprise", usecwd=True), override=False)
