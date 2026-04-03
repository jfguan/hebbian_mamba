from train.configs import ModelConfig, ModelType


def build_model(model_config: ModelConfig):
    if model_config.model == ModelType.DELTA:
        from .delta import Delta
        return Delta(model_config)
    elif model_config.model == ModelType.HYBRID:
        from .hybrid import Hybrid
        return Hybrid(model_config)
    elif model_config.model == ModelType.GDN:
        from .gated_deltanet import GatedDeltaNet
        return GatedDeltaNet(model_config)
    elif model_config.model == ModelType.TRANSFORMER:
        from experimental.transformer import Transformer
        return Transformer(model_config, token_shift=False)
    elif model_config.model == ModelType.TRANSFORMER_TS:
        from experimental.transformer import Transformer
        return Transformer(model_config, token_shift=True)
    elif model_config.model == ModelType.GDN_NOSILU:
        from experimental.gdn_nosilu import GDNNoSiLU
        return GDNNoSiLU(model_config)
    elif model_config.model == ModelType.GDN_TS_STOPGRAD:
        from experimental.gdn_tokenshift_stopgrad import GDNTSStopGrad
        return GDNTSStopGrad(model_config)
    elif model_config.model == ModelType.GDN_TOKENSHIFT:
        from experimental.gdn_tokenshift import GDNTokenShift
        return GDNTokenShift(model_config)
    else:
        raise ValueError(f"Unknown model type: {model_config.model!r}")
