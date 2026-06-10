from pathlib import Path
from pydantic_settings import BaseSettings

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent


class Settings(BaseSettings):
    DATABASE_URL: str

    APP_HOST: str = "127.0.0.1"
    APP_PORT: int = 8000
    DEBUG: bool = False

    CLASSIFIER_MODEL_PATH: str = str(_PROJECT_ROOT / "models" / "efficientnet_b4_classifier" / "efficientnet_b4_classifier.pth")

    SCHEDULER_TIMEOUT_SECONDS: int = 12
    SCHEDULER_MAX_DISPLACEMENT_LAYERS: int = 1
    LAZY_LOADING: bool = True


    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()
