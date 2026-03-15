from dataclasses import asdict

from train.configs import ModelType


def build_model(model_config):
    fields = asdict(model_config)
    fields.pop("name")
    model_type = fields.pop("model")
    fields = {k: v for k, v in fields.items() if v is not None}

    if model_type == ModelType.HEBBIAN:
        from .hebbian_minimal import Config, HebbianConv
        config = Config(**{k: v for k, v in fields.items() if hasattr(Config, k)})
        return HebbianConv(config)
    elif model_type == ModelType.HEBBIAN_MAMBA:
        from .hebbian_mamba import Config, HebbianMamba
        config = Config(**{k: v for k, v in fields.items() if hasattr(Config, k)})
        return HebbianMamba(config)
    elif model_type == ModelType.MAMBA:
        from .mamba import Config, Mamba
        config = Config(**{k: v for k, v in fields.items() if hasattr(Config, k)})
        return Mamba(config)
    else:
        raise ValueError(f"Unknown model type: {model_type!r}")
