from typing import Optional
from pydantic import BaseSettings, Field, BaseModel
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(".env"))

# modified from example here:
# https://rednafi.github.io/digressions/python/2020/06/03/python-configs.html

class AppConfig(BaseModel):
    """Application configurations."""
    app_name: str = "TruffleAI Small Semantic Theo Search"


class GlobalConfig(BaseSettings):
    """Global configurations."""

    # These variables will be loaded from the .env file. However, if
    # there is a shell environment variable having the same name,
    # that will take precedence.

    APP_CONFIG: AppConfig = AppConfig()

    # define global variables with the Field class
    ENV_STATE: Optional[str] = Field('dev', env="ENV_STATE")

    # environment specific variables do not need the Field class
    N_CLOSEST: int = 3
    N_DEPTH: int = 2
    ALGO: str = 'inner'
    USE_MODULE_URL: str = None
    BIBLE_DB_PATH: str = None
    BIBLE_EMBEDDINGS_PATH: str = None
    NETWORKX_GRAPH_DB_PATH: str = None
    
    class Config:
        """Loads the dotenv file."""
        env_file: str = ".env"


class DevConfig(GlobalConfig):
    """Development configurations."""
    class Config:
        env_prefix: str = "DEV_"


class StagingConfig(GlobalConfig):
    """Production configurations."""
    class Config:
        env_prefix: str = "STAGING_"


class ProdConfig(GlobalConfig):
    """Production configurations."""
    class Config:
        env_prefix: str = "PROD_"


class FactoryConfig:
    """Returns a config instance dependending on the ENV_STATE variable."""

    def __init__(self, env_state: Optional[str]):
        self.env_state = env_state

    def __call__(self):
        if self.env_state == "dev":
            return DevConfig()
        elif self.env_state == "staging":
            return StagingConfig()
        elif self.env_state == "prod":
            return ProdConfig()

def cfg_factory():
    return FactoryConfig(GlobalConfig().ENV_STATE)()


# CONFIG = FactoryConfig(GlobalConfig().ENV_STATE)()
# print(CONFIG.__repr__())
