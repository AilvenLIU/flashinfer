"""Cheap auto-admission policy; no artifact loading, JIT, or device work."""

import torch

from ...fused_moe.api import QuantFormat, RoutingInputMode

MIN_AUTO_ROWS = 8192
# Geometries with packaged, validated FC1 and FC2 objects. This only admits a
# candidate; the original backend pool still participates in autotuning.
AUTO_GEOMETRIES = frozenset({(12, 7168, 3072), (8, 4096, 14336)})


def large_bf16_moe(config, act, arch):
    x = act.hidden_states_q
    return (
        arch == 100
        and config.quant.pair == (QuantFormat.BF16, QuantFormat.BF16)
        and config.quant.output == QuantFormat.BF16
        and act.routing_input_mode == RoutingInputMode.PackedPrecomputed
        and x.ndim == 2
        and x.dtype == torch.bfloat16
        and (config.routing.num_experts, x.shape[1], config.experts.intermediate_size)
        in AUTO_GEOMETRIES
        and act.num_tokens * config.routing.top_k >= MIN_AUTO_ROWS
    )
