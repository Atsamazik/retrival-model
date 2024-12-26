import json
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

PROJ_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJ_ROOT / "configurations/configs.json"


class LossFunc(Enum):
    contrastive_loss = 'contrastive_loss'
    triplet_margin_loss = 'triplet_margin_loss'


class Optimizer(Enum):
    adam = 'adam'
    adamw = 'adamw'


class TrainingConfig(BaseModel):
    BASE_MODEL_NAME: str
    learning_rate: float = Field(gt=0.0, lt=1.0)
    seed: int = Field(default=42)
    batch_size: int
    num_workers: int
    optimizer: Optimizer
    loss: LossFunc


def load_config(config_path: Path = CONFIG_PATH) -> TrainingConfig:
    try:
        with open(config_path, "r") as file:
            data = json.load(file)

        configuration = TrainingConfig(**data)
        return configuration
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {config_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in config file: {config_path}")


if __name__ == '__main__':
    config = load_config()
    print(config.BASE_MODEL_NAME)
    print(config.learning_rate)
    print(config.seed)