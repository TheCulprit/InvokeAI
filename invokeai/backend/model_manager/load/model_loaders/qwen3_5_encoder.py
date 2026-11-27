"""Loading of Qwen3.5 text encoders.

`load_qwen3_5_text_encoder` builds the encoder from the language-model tensors of one safetensors
file or of a model's shards; the single-file loader below and a model folder's loader both call it.
"""

from collections.abc import Iterable, Sequence
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import accelerate
import torch
from safetensors import safe_open
from torch import nn

from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.configs.qwen3_5_encoder import (
    QWEN3_5_FINAL_NORM_VARIANTS,
    Qwen35Encoder_Checkpoint_Config,
    Qwen35Encoder_GGUF_Config,
    qwen3_5_key_prefix,
)
from invokeai.backend.model_manager.load.load_default import ModelLoader, _model_declared_skip_patterns
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.taxonomy import AnyModel, BaseModelType, ModelFormat, ModelType, SubModelType
from invokeai.backend.model_manager.util.llamacpp_keys import convert_llamacpp_decoder_keys
from invokeai.backend.quantization.fp8_scaled import cast_state_dict, reject_quantized_side_channel, safetensors_backend
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor
from invokeai.backend.quantization.gguf.loaders import gguf_sd_loader, unpack_ggml_at_load
from invokeai.backend.quantization.gguf.row_ops import split_rows, take_rows
from invokeai.backend.quantization.int8_convrot import (
    drop_unconsumed_quantization_sidecars,
    extract_int8_convrot_markers,
    install_int8_convrot_layers,
    reject_unmarked_int8_weights,
)
from invokeai.backend.quantization.load_plan import reserve_for_load
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.state_dict_loading import log_unexpected_keys, reject_incomplete_load

if TYPE_CHECKING:
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    from invokeai.backend.qwen3_5.qwen3_5_encoder import Qwen35Encoder


class _ZeroMLP(nn.Module):
    """Stands in for an MLP sub-block the checkpoint does not ship, so the layer adds nothing there."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


def _encoder_names(keys: Iterable[str], prefix: str, *, final_norm: bool) -> dict[str, str]:
    """The checkpoint keys `Qwen35Encoder` holds, mapped to its names for them.

    Everything else stays on disk: a multimodal export's vision tower, the LM head, the MTP head and,
    unless the final norm is read, whatever sits under `norm.*` -- in Anima-3.8B's
    `qwen35_4b.safetensors`, a 2560->1024 projection head that no encoder of it runs.
    """
    names: dict[str, str] = {}
    for key in keys:
        name = key[len(prefix) :]
        if key.startswith(prefix) and (
            name.startswith(("embed_tokens.", "layers.")) or (final_norm and name == "norm.weight")
        ):
            names[key] = name
    return names


def load_qwen3_5_text_encoder(
    loader: ModelLoader,
    files: Sequence[Path],
    text_config: "Qwen3_5TextConfig",
    *,
    final_norm: bool,
    keep_fp8: bool,
    what: str,
) -> "Qwen35Encoder":
    """Build a `Qwen35Encoder` from the language-model tensors of a safetensors file or of a model's shards.

    Nothing else is read, so a multimodal export's vision tower, an LM head and a multi-token-prediction
    head never reach RAM. A ComfyUI `int8_tensorwise` build keeps its marked Linears int8, dequantizing
    per forward; the embedding, which is not a Linear, is widened on load.

    Args:
        loader: The calling loader; the load reserves room in its RAM cache.
        files: The safetensors file, or every shard of the model.
        text_config: The architecture to build.
        final_norm: Build and load the final norm, for a caller that reads the last hidden state. The
            checkpoint must then ship `norm.weight` and every layer whole.
        keep_fp8: Whether a raw-fp8 checkpoint's weights stay packed rather than widened: the calling
            loader's `_keep_fp8_weights` answer, asked by the caller because `test_fp8_capability.py`
            can only see the fp8 helpers reached through a loader's own methods. For an encoder that
            means the fp8 matmul: text encoders never take FP8 Storage's layerwise cast
            (`_FP8_EXCLUDED_SUBMODEL_TYPES`).
        what: Names the checkpoint in errors and log lines.
    """
    from invokeai.backend.qwen3_5.qwen3_5_encoder import Qwen35Encoder

    keys_by_file: dict[Path, list[str]] = {}
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as f:
            keys_by_file[path] = list(f.keys())
    all_keys = dict.fromkeys(key for keys in keys_by_file.values() for key in keys)
    prefix = qwen3_5_key_prefix(all_keys)
    if prefix is None:
        raise ValueError(f"{what} is not a Qwen3.5 language model checkpoint.")
    names = _encoder_names(all_keys, prefix, final_norm=final_norm)

    sd: dict[str, torch.Tensor] = {}
    for path, keys in keys_by_file.items():
        with safe_open(path, framework="pt", device="cpu", backend=safetensors_backend(path)) as f:
            for key in keys:
                if key in names:
                    sd[names[key]] = f.get_tensor(key)

    with accelerate.init_empty_weights():
        model = Qwen35Encoder(text_config, final_norm=final_norm)

    if not final_norm:
        # Anima-3.8B's encoder ships its last layer without the MLP sub-block (and its norm), because
        # nothing it was used for reads past that layer's attention. Give such a layer an MLP that adds
        # nothing, so the layer computes exactly what the file encodes instead of loading half-empty.
        # A caller of the final norm runs every layer whole, so there a missing MLP stays a missing tensor.
        for index, layer in enumerate(model.layers):
            if not any(key.startswith(f"layers.{index}.mlp.") for key in sd):
                layer.mlp = _ZeroMLP()
                layer.post_attention_layernorm = nn.Identity()
                loader._logger.info(f"Qwen3.5 encoder: layer {index} ships without an MLP; running it attention-only.")

    model_dtype = TorchDevice.choose_bfloat16_safe_dtype(TorchDevice.choose_torch_device())
    skip_patterns = _model_declared_skip_patterns(model)
    markers = extract_int8_convrot_markers(sd)
    # Outside the int8 branch, so an int8 weight whose marker is missing is refused, not cast as codes.
    reject_unmarked_int8_weights(sd, markers, "Qwen3.5", model)
    kept = 0
    if markers:
        sd = drop_unconsumed_quantization_sidecars(sd)
        int8_layers = install_int8_convrot_layers(
            model,
            sd,
            markers,
            model_dtype,
            architecture="Qwen3.5",
            reserve=loader.ram_cache.make_room,
            skip_patterns=skip_patterns,
        )
        loader._logger.info(
            f"{what}: kept {len(int8_layers)} of {len(markers)} layer(s) in int8 (int8_tensorwise checkpoint, "
            "dequantized per forward); the embedding is widened on load."
        )
    else:
        # Anima-3.8B's encoder ships raw fp8 projections (no scale), which the cast below handles
        # exactly. A scaled or nvfp4 build would load *wrong* here rather than fail, so refuse it.
        reject_quantized_side_channel(sd, what)
        reserve_for_load(
            loader.ram_cache.make_room,
            sd,
            model_dtype,
            keep_fp8=keep_fp8,
            model=model,
            skip_patterns=skip_patterns,
            fp8_layers={},
            nvfp4_payloads={},
        )
        kept = cast_state_dict(sd, model_dtype, keep_fp8=keep_fp8, model=model, skip_patterns=skip_patterns)

    result = model.load_state_dict(sd, strict=False, assign=True)
    log_unexpected_keys(what, result.unexpected_keys)
    reject_incomplete_load(model, what=what)
    sd.clear()

    if kept:
        loader._logger.info(f"Qwen3.5 encoder: kept {kept} raw fp8 weight(s) quantized ({loader._fp8_kept_reason()}).")
    return model.eval()


@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.Qwen35Encoder, format=ModelFormat.Checkpoint)
class Qwen35EncoderCheckpointLoader(ModelLoader):
    """Loads a single-file Qwen3.5 text encoder and the vendored Qwen3.5 tokenizer."""

    def get_size_fs(
        self, config: AnyModelConfig, model_path: Path, submodel_type: Optional[SubModelType] = None
    ) -> int:
        # A few small tokenizer files, or the vendored one: the generic sizing would charge the whole model.
        if submodel_type is SubModelType.Tokenizer:
            return 0
        return super().get_size_fs(config, model_path, submodel_type)

    def _load_model(self, config: AnyModelConfig, submodel_type: Optional[SubModelType] = None) -> AnyModel:
        if not isinstance(config, Qwen35Encoder_Checkpoint_Config):
            raise ValueError("Only Qwen35Encoder_Checkpoint_Config models are supported here.")

        match submodel_type:
            case SubModelType.TextEncoder:
                return self._load_text_encoder(config)
            case SubModelType.Tokenizer:
                from invokeai.backend.qwen3_5.qwen3_5_encoder import load_bundled_qwen3_5_tokenizer

                # Single-file checkpoints ship no tokenizer. The vendored one serves the 4B and the 9B.
                return load_bundled_qwen3_5_tokenizer()

        raise ValueError(
            f"Only TextEncoder and Tokenizer submodels are supported. Received: {submodel_type.value if submodel_type else 'None'}"
        )

    def _load_text_encoder(self, config: Qwen35Encoder_Checkpoint_Config) -> AnyModel:
        from invokeai.backend.qwen3_5.qwen3_5_encoder import qwen3_5_text_config

        model_path = Path(config.path)
        model = load_qwen3_5_text_encoder(
            self,
            [model_path],
            qwen3_5_text_config(config.variant),
            # MOSS-Image reads the 9B after its final norm. Anima-3.8B reads intermediate layers of the 4B,
            # and its checkpoint keeps a projection head under `norm.*` instead.
            final_norm=config.variant in QWEN3_5_FINAL_NORM_VARIANTS,
            keep_fp8=self._keep_fp8_weights(config, SubModelType.TextEncoder),
            what=f"Qwen3.5 encoder checkpoint {model_path.name}",
        )
        return self._apply_fp8_layerwise_casting(model, config, SubModelType.TextEncoder)


_GGUF_COMPONENTS = {
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "attn_qkv.weight": "linear_attn.in_proj_qkv.weight",
    "attn_gate.weight": "linear_attn.in_proj_z.weight",
    "ssm_alpha.weight": "linear_attn.in_proj_a.weight",
    "ssm_beta.weight": "linear_attn.in_proj_b.weight",
    "ssm_a": "linear_attn.A_log",
    "ssm_dt.bias": "linear_attn.dt_bias",
    "ssm_conv1d.weight": "linear_attn.conv1d.weight",
    "ssm_norm.weight": "linear_attn.norm.weight",
    "ssm_out.weight": "linear_attn.out_proj.weight",
}


def _tile_output_projection_input(
    _module: nn.Module,
    inputs: tuple[torch.Tensor],
    *,
    key_heads: int,
    values_per_key: int,
    value_head_size: int,
) -> tuple[torch.Tensor]:
    (hidden_states,) = inputs
    tiled = hidden_states.unflatten(-1, (key_heads, values_per_key, value_head_size)).transpose(-3, -2)
    return (tiled.reshape_as(hidden_states),)


def _restore_gguf_layout(raw: dict[str, torch.Tensor], model: "Qwen35Encoder") -> dict[str, torch.Tensor]:
    """Reverse llama.cpp's Qwen3.5 tensor conversions while keeping Linear weights packed."""
    config = model.config
    key_heads = config.linear_num_key_heads
    value_heads = config.linear_num_value_heads
    values_per_key = value_heads // key_heads
    value_head_size = config.linear_value_head_dim
    qk_rows = 2 * key_heads * config.linear_key_head_dim
    value_rows = value_heads * value_head_size

    # llama.cpp tiles V heads to broadcast K across them. Transformers repeats each K head
    # consecutively, so restore the corresponding groups of V rows in the packed weights.
    head_order = torch.arange(value_heads).reshape(values_per_key, key_heads).T.flatten()
    value_order = (head_order[:, None] * value_head_size + torch.arange(value_head_size)).flatten()
    qkv_order = torch.cat((torch.arange(qk_rows), qk_rows + value_order))
    tile_input = partial(
        _tile_output_projection_input,
        key_heads=key_heads,
        values_per_key=values_per_key,
        value_head_size=value_head_size,
    )

    named = convert_llamacpp_decoder_keys(raw)
    raw.clear()
    for index, layer in enumerate(model.layers):
        if config.layer_types[index] != "linear_attention":
            continue
        prefix = f"model.layers.{index}."
        for fused, left, right, rows in (
            ("attn_qkv_gate", "attn_qkv", "attn_gate", qk_rows + value_rows),
            ("ssm_beta_alpha", "ssm_beta", "ssm_alpha", value_heads),
        ):
            if (weight := named.pop(f"{prefix}{fused}.weight", None)) is not None:
                named[f"{prefix}{left}.weight"], named[f"{prefix}{right}.weight"] = split_rows(weight, rows)

        # Reorder the activation instead of the columns of the output projection. Its GGML blocks
        # span those columns, so a weight permutation could require unpacking and requantizing.
        layer.linear_attn.out_proj.register_forward_pre_hook(tile_input)

    sd: dict[str, torch.Tensor] = {}
    while named:
        key, weight = named.popitem()
        name = key.removeprefix("model.")
        if name.startswith("layers."):
            layers, index, component = name.split(".", 2)
            name = f"{layers}.{index}.{_GGUF_COMPONENTS.get(component, component)}"
        if name.endswith(("linear_attn.conv1d.weight", "linear_attn.in_proj_qkv.weight")):
            weight = take_rows(weight, qkv_order)
        elif name.endswith("linear_attn.in_proj_z.weight"):
            weight = take_rows(weight, value_order)
        elif name.endswith(("linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight")):
            weight = take_rows(weight, head_order)
        elif name.endswith(("linear_attn.A_log", "linear_attn.dt_bias")):
            # Recover the decay parameter before lowering precision. The HF forward exponentiates it.
            weight = _gguf_float32(weight)[head_order]
            if name.endswith(".A_log"):
                weight = (-weight).log()
        elif name.endswith("norm.weight") and not name.endswith("linear_attn.norm.weight"):
            # Transformers's RMSNorm adds 1 in forward; llama.cpp folds that offset into the file.
            weight = _gguf_float32(weight) - 1
        sd[name] = weight
    return sd


def _gguf_float32(weight: torch.Tensor) -> torch.Tensor:
    if isinstance(weight, GGMLTensor):
        weight.compute_dtype = torch.float32
        return weight.get_dequantized_tensor()
    return weight.float()


@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.Qwen35Encoder, format=ModelFormat.GGUFQuantized)
class Qwen35EncoderGGUFLoader(ModelLoader):
    """Load the Qwen3.5 9B language tower as an encoder with Invoke's bundled tokenizer."""

    def get_size_fs(
        self, config: AnyModelConfig, model_path: Path, submodel_type: Optional[SubModelType] = None
    ) -> int:
        if submodel_type is SubModelType.Tokenizer:
            return 0
        return super().get_size_fs(config, model_path, submodel_type)

    def _load_model(self, config: AnyModelConfig, submodel_type: Optional[SubModelType] = None) -> AnyModel:
        if not isinstance(config, Qwen35Encoder_GGUF_Config):
            raise TypeError(f"Expected Qwen35Encoder_GGUF_Config, got {type(config).__name__}.")
        if submodel_type is SubModelType.Tokenizer:
            from invokeai.backend.qwen3_5.qwen3_5_encoder import load_bundled_qwen3_5_tokenizer

            return load_bundled_qwen3_5_tokenizer()
        if submodel_type is SubModelType.TextEncoder:
            return self._load_text_encoder(config)
        raise ValueError(f"Expected a Qwen3.5 GGUF TextEncoder or Tokenizer, got {submodel_type}.")

    def _load_text_encoder(self, config: Qwen35Encoder_GGUF_Config) -> "Qwen35Encoder":
        from invokeai.backend.qwen3_5.qwen3_5_encoder import Qwen35Encoder, qwen3_5_text_config

        text_config = qwen3_5_text_config(config.variant)
        with accelerate.init_empty_weights():
            model = Qwen35Encoder(text_config, final_norm=True)
        model_dtype = TorchDevice.choose_bfloat16_safe_dtype(TorchDevice.choose_torch_device())
        layers = tuple(f"blk.{index}." for index in range(text_config.num_hidden_layers))
        raw = gguf_sd_loader(
            Path(config.path),
            compute_dtype=model_dtype,
            keep=lambda name: name in {"token_embd.weight", "output_norm.weight"} or name.startswith(layers),
        )
        sd = _restore_gguf_layout(raw, model)
        unpack_ggml_at_load(sd, model, _model_declared_skip_patterns(model), self.ram_cache.make_room)
        for index, layer_type in enumerate(text_config.layer_types):
            if layer_type == "linear_attention":
                key = f"layers.{index}.linear_attn.conv1d.weight"
                if key in sd:
                    sd[key] = sd[key].unsqueeze(1)
        what = f"Qwen3.5 GGUF encoder {config.name!r}"
        result = model.load_state_dict(sd, strict=False, assign=True)
        log_unexpected_keys(what, result.unexpected_keys)
        reject_incomplete_load(model, what=what)
        return model.eval()
