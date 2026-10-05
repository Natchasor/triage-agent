"""Application settings, loaded from environment variables."""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "gpt-5.6-terra"


@dataclass(frozen=True)
class Settings:
    model: str = DEFAULT_MODEL
    data_dir: Path = PROJECT_ROOT / "data"
    max_tool_rounds: int = 6

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(PROJECT_ROOT / ".env")
        return cls(model=os.getenv("OPENAI_MODEL", DEFAULT_MODEL))
