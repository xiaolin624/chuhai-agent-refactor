from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """全局配置，从环境变量 / .env 文件读取。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── 应用基础 ──
    APP_NAME: str = "chuhai-agent"
    APP_VERSION: str = "0.1.0"
    ENV: str = "development"
    DEV_MODE: bool = False

    # ── 数据库 ──
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/chuhai_agent"

    # ── CORS ──
    CORS_ORIGINS: str = "*"

    # ── 鉴权 ──
    JWT_SECRET_KEY: str = ""
    JWT_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 天
    WECHAT_APP_ID: str = ""
    WECHAT_APP_SECRET: str = ""
    WECHAT_REDIRECT_URI: str = ""
    WECHAT_QR_URL: str = ""

    # ── Admin ──
    ADMIN_API_KEY: str = ""

    # ── AI ──
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_MODEL: str = "deepseek-v4-flash"
    DEEPSEEK_EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_API_KEY: str = ""
    EMBEDDING_BASE_URL: str = ""
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"

    # ── Agent Runtime ──
    DIALOGUE_MAX_TOKENS: int = 1024
    DIALOGUE_TEMPERATURE: float = 0.2
    DIALOGUE_HISTORY_WINDOW: int = 12
    REPORT_MAX_TOKENS: int = 4000
    REPORT_ESCALATED_MAX_TOKENS: int = 8000
    MAX_AGENT_STEPS: int = 16

    # ── 编排器 ──
    # handwritten（默认）| langgraph
    # langgraph 走 app/agent/graph_lg.py 的事件路由图；未安装 langgraph 时自动降级为手写。
    AGENT_ORCHESTRATOR: str = "handwritten"


@lru_cache
def get_settings() -> Settings:
    """返回全局唯一的 Settings 实例（lru_cache 保证单例）。"""
    return Settings()
