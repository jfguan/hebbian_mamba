from dataclasses import dataclass
from enum import Enum

from data.loader import DatasetName


class ModelType(str, Enum):
    HEBBIAN = "hebbian"
    HEBBIAN_MAMBA = "hebbian_mamba"
    MAMBA = "mamba"


@dataclass
class ModelConfig:
    name: str
    model: ModelType
    d_model: int
    n_layers: int
    d_conv: int
    expand: int
    d_state: int
    # Hebbian memory (optional, only for hebbian models)
    memory_alpha: float | None = None
    chunk_size: int | None = None


@dataclass
class TrainConfig:
    dataset: DatasetName
    steps: int
    batch_size: int
    seq_len: int
    lr: float
    warmup: int
    grad_accum: int
    eval_interval: int
    ckpt_interval: int


# -- Hebbian --

HEBBIAN_18M = ModelConfig(
    name="hebbian_18M",
    model=ModelType.HEBBIAN,
    d_model=512,
    n_layers=8,
    d_conv=4,
    expand=2,
    d_state=16,
    memory_alpha=0.03,
    chunk_size=64,
)

HEBBIAN_100M = ModelConfig(
    name="hebbian_100M",
    model=ModelType.HEBBIAN,
    d_model=1024,
    n_layers=12,
    d_conv=4,
    expand=2,
    d_state=16,
    memory_alpha=0.03,
    chunk_size=64,
)

# -- Hebbian mamba --

HEBBIAN_MAMBA_18M = ModelConfig(
    name="hebbian_mamba_18M",
    model=ModelType.HEBBIAN_MAMBA,
    d_model=512,
    n_layers=8,
    d_conv=4,
    expand=2,
    d_state=16,
    memory_alpha=0.03,
    chunk_size=64,
)

HEBBIAN_MAMBA_100M = ModelConfig(
    name="hebbian_mamba_100M",
    model=ModelType.HEBBIAN_MAMBA,
    d_model=1024,
    n_layers=12,
    d_conv=4,
    expand=2,
    d_state=16,
    memory_alpha=0.03,
    chunk_size=64,
)

# -- Mamba baseline --

MAMBA_100M = ModelConfig(
    name="mamba_100M",
    model=ModelType.MAMBA,
    d_model=1024,
    n_layers=16,
    d_conv=4,
    expand=2,
    d_state=16,
)


# -- Training hyperparameters --

TRAIN_STACK_18M = TrainConfig(
    dataset=DatasetName.THE_STACK,
    steps=1221,
    batch_size=4,
    seq_len=2048,
    lr=6e-4,
    warmup=20,
    grad_accum=1,
    eval_interval=100,
    ckpt_interval=1221,
)

TRAIN_STACK_100M = TrainConfig(
    dataset=DatasetName.THE_STACK,
    steps=64000,
    batch_size=1,
    seq_len=2048,
    lr=3e-4,
    warmup=500,
    grad_accum=1,
    eval_interval=200,
    ckpt_interval=64000,
)
