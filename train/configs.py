from dataclasses import dataclass
from enum import Enum

from data.loader import DatasetName


class ModelType(str, Enum):
    DELTA = "delta"
    HYBRID = "hybrid"
    GDN = "gdn"
    GDN_TOKENSHIFT = "gdn_tokenshift"
    GDN_NOSILU = "gdn_nosilu"
    TRANSFORMER = "transformer"
    TRANSFORMER_TS = "transformer_ts"
    GDN_TS_STOPGRAD = "gdn_ts_stopgrad"


@dataclass
class ModelConfig:
    name: str
    model: ModelType
    d_model: int
    n_layers: int
    d_conv: int
    expand: int
    d_state: int
    chunk_size: int

    vocab_size: int = 0  # set from dataset at runtime

    # Hebbian memory
    num_heads: int | None = None  # memory heads (None or 1 = full D×D)

    # Delta Hebbian
    delta_layers: list[int] | None = None  # layer indices that use delta rule
    delta_num_heads: int | None = None  # heads for delta layers (default: 8)
    delta_num_matrices: int = 1  # state matrices per delta block (1-4)

    # SWA
    swa_window: int = 256  # sliding window attention size

    def __post_init__(self):
        # migrate old fields from checkpoints
        for attr in (
            "head_dim",
            "delta_head_dim",
            "memory_alpha",
            "neg_eigenvalues",
            "no_memory_layers",
        ):
            if hasattr(self, attr):
                delattr(self, attr)
        if isinstance(self.delta_layers, str):
            self.delta_layers = [int(x) for x in self.delta_layers.split(",")]


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
    max_steps_per_run: int | None = None
    compile: bool = False


# -- Model configs (name is overridden at runtime to <model>_<dataset>_<size>) --

DELTA_18M = ModelConfig(
    name="delta",
    model=ModelType.DELTA,
    d_model=512,
    n_layers=7,
    d_conv=4,
    expand=2,
    d_state=16,
    chunk_size=64,
    delta_num_heads=4,
    delta_layers=list(range(7)),
)

DELTA_100M = ModelConfig(
    name="delta",
    model=ModelType.DELTA,
    d_model=1024,
    n_layers=12,
    d_conv=4,
    expand=2,
    d_state=16,
    chunk_size=32,
    delta_num_heads=8,
    delta_layers=list(range(12)),
)

HYBRID_18M = ModelConfig(
    name="hybrid",
    model=ModelType.HYBRID,
    d_model=512,
    n_layers=8,
    d_conv=4,
    expand=2,
    d_state=16,
    chunk_size=64,
    num_heads=4,
    delta_num_heads=4,
    delta_layers=[6, 7],
    swa_window=256,
)

HYBRID_100M = ModelConfig(
    name="hybrid",
    model=ModelType.HYBRID,
    d_model=1024,
    n_layers=12,
    d_conv=4,
    expand=2,
    d_state=16,
    chunk_size=64,
    num_heads=8,
    delta_num_heads=8,
    delta_layers=[3, 7, 11],
)

GDN_18M = ModelConfig(
    name="gdn",
    model=ModelType.GDN,
    d_model=512,
    n_layers=6,
    d_conv=4,
    expand=2,
    d_state=16,
    chunk_size=64,
    num_heads=4,
)

GDN_TS_18M = ModelConfig(
    name="gdn_tokenshift",
    model=ModelType.GDN_TOKENSHIFT,
    d_model=512,
    n_layers=6,
    d_conv=4,
    expand=2,
    d_state=16,
    chunk_size=64,
    num_heads=4,
)

GDN_TS_SG_18M = ModelConfig(
    name="gdn_ts_stopgrad",
    model=ModelType.GDN_TS_STOPGRAD,
    d_model=512,
    n_layers=6,
    d_conv=4,
    expand=2,
    d_state=16,
    chunk_size=64,
    num_heads=4,
)

GDN_100M = ModelConfig(
    name="gdn",
    model=ModelType.GDN,
    d_model=1024,
    n_layers=9,
    d_conv=4,
    expand=2,
    d_state=16,
    chunk_size=64,
    num_heads=8,
)

GDN_TS_100M = ModelConfig(
    name="gdn_tokenshift",
    model=ModelType.GDN_TOKENSHIFT,
    d_model=1024,
    n_layers=9,
    d_conv=4,
    expand=2,
    d_state=16,
    chunk_size=64,
    num_heads=8,
)

TRANSFORMER_18M = ModelConfig(
    name="transformer",
    model=ModelType.TRANSFORMER,
    d_model=512,
    n_layers=8,
    d_conv=4,
    expand=2,
    d_state=16,
    chunk_size=64,
    num_heads=4,
)

TRANSFORMER_TS_18M = ModelConfig(
    name="transformer_ts",
    model=ModelType.TRANSFORMER_TS,
    d_model=512,
    n_layers=8,
    d_conv=4,
    expand=2,
    d_state=16,
    chunk_size=64,
    num_heads=4,
)

GDN_NOSILU_18M = ModelConfig(
    name="gdn_nosilu",
    model=ModelType.GDN_NOSILU,
    d_model=512,
    n_layers=6,
    d_conv=4,
    expand=2,
    d_state=16,
    chunk_size=64,
    num_heads=4,
)


# -- Training hyperparameters --

TRAIN_PG19_18M = TrainConfig(
    dataset=DatasetName.PG19,
    steps=1221,
    batch_size=4,
    seq_len=2048,
    lr=6e-4,
    warmup=60,
    grad_accum=1,
    eval_interval=100,
    ckpt_interval=1221,
    max_steps_per_run=1221,
)

TRAIN_STACK_18M = TrainConfig(
    dataset=DatasetName.THE_STACK,
    steps=3662,  # ~30M tokens (was 1221 for ~10M)
    batch_size=4,
    seq_len=2048,
    lr=6e-4,
    warmup=180,
    grad_accum=1,
    eval_interval=200,
    ckpt_interval=3662,
    max_steps_per_run=3662,
)

TRAIN_STACK_100M = TrainConfig(
    dataset=DatasetName.THE_STACK,
    steps=244000,
    batch_size=1,
    seq_len=2048,
    lr=3e-4,
    warmup=5000,
    grad_accum=1,
    eval_interval=1000,
    ckpt_interval=10000,
    max_steps_per_run=48828,  # 48828 = 100M tokens, adjusted for delta resume at 60K
)
