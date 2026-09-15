"""Internal BF16 FC2 PoC: prepared execution of offline grouped GEMM objects.

This is not a public MoE API. Inputs are already grouped, with one group per
local expert. Output is unweighted BF16; routing weights and final scatter
belong to the surrounding MoE pipeline. No Frost compiler is imported here.
"""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .runtime import (
    _arch_for,
    _artifact_roots,
    _current_custream,
    _digest,
    _dimension_matches,
    _load_object,
    _safe_child,
)


@dataclass(frozen=True)
class Fc2Kernel:
    artifact_id: str
    arch: str
    symbol: str
    object_path: Path
    object_sha256: str
    workspace_bytes: int
    contract: dict[str, Any]
    tactic_metadata: dict[str, Any]

    @property
    def tactic(self) -> tuple[str, str, str]:
        return ("frost-grouped-fc2-v1", self.artifact_id, self.object_sha256[:20])


@functools.lru_cache(maxsize=8)
def discover(roots: tuple[Path, ...] | None = None) -> tuple[Fc2Kernel, ...]:
    """Read and verify sealed FC2 artifacts, separately from the FC1 pool."""
    result = []
    seen = set()
    for root in _artifact_roots() if roots is None else roots:
        path = root / "frost_selected_kernels.json"
        payload = json.loads(path.read_text())
        if payload.get("schema_version") != 1 or not isinstance(
            payload.get("kernels"), list
        ):
            raise RuntimeError(f"invalid FROST manifest: {path}")
        for raw in payload["kernels"]:
            if raw.get("op") != "grouped_gemm2":
                continue
            identity = raw.get("id")
            if not isinstance(identity, str) or not identity or identity in seen:
                raise RuntimeError("FC2 artifact ids must be non-empty and unique")
            seen.add(identity)
            if raw.get("abi") != "frost_grouped_gemm2_v1" or raw.get("launch") != {
                "tail": ["output"]
            }:
                raise RuntimeError(f"unsupported FC2 ABI: {identity}")
            obj = raw["object"]
            object_path = _safe_child(root, obj["path"])
            digest = _digest(object_path)
            if digest != obj.get("sha256"):
                raise RuntimeError(f"FC2 object digest mismatch: {object_path}")
            size = raw["workspace_bytes"]
            if not isinstance(size, int) or size < 0 or size % 128:
                raise RuntimeError(
                    "FC2 workspace must be a nonnegative multiple of 128"
                )
            if not isinstance(raw.get("symbol"), str) or not raw["symbol"]:
                raise RuntimeError("FC2 artifact requires an exported symbol")
            contract = raw["contract"]
            if (
                any(
                    contract.get(key) != "bfloat16"
                    for key in ("token_dtype", "weight_dtype", "output_dtype")
                )
                or contract.get("activation") != "identity"
            ):
                raise RuntimeError(
                    "FC2 v1 requires BF16 inputs/output and identity activation"
                )
            result.append(
                Fc2Kernel(
                    identity,
                    raw["arch"],
                    raw["symbol"],
                    object_path,
                    digest,
                    size,
                    contract,
                    raw["tactic"],
                )
            )
    return tuple(result)


def matching_kernels(
    rows: int, hidden: int, intermediate: int, experts: int, device: torch.device
) -> tuple[Fc2Kernel, ...]:
    values = dict(s=rows, n=hidden, k=intermediate, experts=experts, groups=experts)
    return tuple(
        kernel
        for kernel in discover()
        if kernel.arch == _arch_for(device)
        and all(
            _dimension_matches(value, kernel.contract.get(key))
            for key, value in values.items()
        )
    )


class PreparedFc2:
    """Bind caller-owned tensors outside capture, then launch without allocation.

    Offsets are int32[E], start at zero, are nondecreasing, and end implicitly
    at S. Empty experts are allowed. Callers may change tensor contents between
    runs, but must preserve valid offsets and keep this plan alive for graphs.
    Each concurrent stream needs its own workspace/plan.
    """

    def __init__(
        self,
        kernel: Fc2Kernel,
        x: torch.Tensor,
        weights: torch.Tensor,
        offsets: torch.Tensor,
        out: torch.Tensor,
        workspace: torch.Tensor | None = None,
    ):
        if x.device.type != "cuda":
            raise ValueError("FC2 requires CUDA tensors")
        with torch.cuda.device(x.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("prepare FC2 outside CUDA Graph capture")
        if x.ndim != 2 or weights.ndim != 3:
            raise ValueError("FC2 expects x[S,I], weights[E,H,I]")
        s, k = x.shape
        e, n, wk = weights.shape
        if wk != k or tuple(out.shape) != (s, n) or tuple(offsets.shape) != (e,):
            raise ValueError("FC2 tensor geometry mismatch")
        values = dict(s=s, n=n, k=k, experts=e, groups=e)
        if kernel.arch != _arch_for(x.device) or not all(
            _dimension_matches(value, kernel.contract.get(key))
            for key, value in values.items()
        ):
            raise ValueError("FC2 artifact geometry/architecture mismatch")
        if min(s, n, k, e) <= 0 or s > 2**31 - 1:
            raise ValueError("FC2 requires positive dimensions and int32 row offsets")
        if workspace is None:
            workspace = torch.empty(
                kernel.workspace_bytes, dtype=torch.uint8, device=x.device
            )
        tensors = (x, weights, offsets, out, workspace)
        if any(t.device != x.device or not t.is_contiguous() for t in tensors):
            raise ValueError("FC2 tensors must be contiguous on one CUDA device")
        if (
            any(t.dtype != torch.bfloat16 for t in (x, weights, out))
            or offsets.dtype != torch.int32
        ):
            raise ValueError("FC2 requires BF16 data and int32 offsets")
        if workspace.dtype != torch.uint8 or workspace.numel() < kernel.workspace_bytes:
            raise ValueError("FC2 workspace is too small or not uint8")
        if workspace.data_ptr() % 128 or any(
            t.data_ptr() % 16 for t in (x, weights, out)
        ):
            raise ValueError("FC2 requires aligned workspace (128B) and data (16B)")
        if len({t.untyped_storage().data_ptr() for t in tensors}) != len(tensors):
            raise ValueError("FC2 input, output and workspace storage must be distinct")
        starts = (
            offsets.tolist()
        )  # Preparation only; never read device offsets at launch.
        if (
            starts[0] != 0
            or any(a > b for a, b in zip(starts[:-1], starts[1:], strict=True))
            or starts[-1] > s
        ):
            raise ValueError("FC2 offsets must start at zero and increase within [0,S]")
        self.kernel = kernel
        self.output = out
        self.device = x.device
        self.workspace = workspace[: kernel.workspace_bytes]
        self._launch = _load_object(
            kernel.object_path, kernel.object_sha256, kernel.symbol
        )
        token = x.unsqueeze(0).permute(1, 2, 0)
        weight = weights.permute(1, 2, 0)
        output = out.unsqueeze(0).permute(1, 2, 0)
        problem = (s, n, k, e, e, *token.stride(), *weight.stride(), *output.stride())
        self._args = (
            problem,
            offsets,
            self.workspace.view(torch.int64),
            token,
            weight,
            output,
        )

    def run(self) -> torch.Tensor:
        # The exported host entry does not reset the persistent scheduler.
        # Both reset and launch use the caller's current stream on this device.
        with torch.cuda.device(self.device):
            self.workspace.zero_()
            self._launch(*self._args, _current_custream(self.device))
        return self.output
