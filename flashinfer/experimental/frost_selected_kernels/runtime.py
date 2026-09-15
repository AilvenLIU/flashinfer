"""Runtime for FROST-generated grouped GEMM1 + SwiGLU kernels.

FROST is used only on the build box. The deployed process validates a sealed
manifest, reloads the exported TVM-FFI object, and marshals the already-grouped
MoE tensors into FROST's compiled host ABI.
"""

from __future__ import annotations

import functools
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ...autotuner import OptimizationProfile, TunableRunner

_MANIFEST = "frost_selected_kernels.json"
_OP = "grouped_gemm1_swiglu"
_ABI = "frost_grouped_gemm1_swiglu_v1"


@dataclass(frozen=True)
class FrostGroupedGemm1SwiGLUKernel:
    artifact_id: str
    arch: str
    symbol: str
    object_path: Path
    object_sha256: str
    workspace_bytes: int
    contract: dict[str, Any]
    tactic_metadata: dict[str, Any]
    launch_tail: tuple[str, str]

    @property
    def tactic(self) -> tuple[str, str, str]:
        """Stable identity suitable for FlashInfer's persisted autotuner."""

        return ("frost-grouped-swiglu-v1", self.artifact_id, self.object_sha256[:20])


def _artifact_roots() -> tuple[Path, ...]:
    packaged = Path(__file__).resolve().parent / "artifacts"
    return (packaged,) if packaged.is_dir() else ()


def _safe_child(root: Path, relative: str) -> Path:
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise RuntimeError(f"invalid FROST artifact path {relative!r}")
    path = (root / rel).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError(f"FROST artifact escapes its root: {relative!r}") from exc
    return path


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_root(root: Path) -> list[FrostGroupedGemm1SwiGLUKernel]:
    path = root / _MANIFEST
    if not path.is_file():
        return []
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 1:
        raise RuntimeError(f"{path}: unsupported FROST manifest schema")
    kernels = payload.get("kernels")
    if not isinstance(kernels, list):
        raise RuntimeError(f"{path}: 'kernels' must be a list")

    result: list[FrostGroupedGemm1SwiGLUKernel] = []
    seen: set[str] = set()
    for raw in kernels:
        artifact_id = raw.get("id")
        if not isinstance(artifact_id, str) or not artifact_id or artifact_id in seen:
            raise RuntimeError(f"{path}: kernel ids must be non-empty and unique")
        seen.add(artifact_id)
        if raw.get("op") != _OP or raw.get("abi") != _ABI:
            continue
        symbol = raw.get("symbol")
        if not isinstance(symbol, str) or not symbol:
            raise RuntimeError(f"{path}: kernel {artifact_id!r} has no symbol")
        obj = raw.get("object", {})
        object_path = _safe_child(root, obj.get("path", ""))
        expected = obj.get("sha256")
        if not object_path.is_file():
            raise FileNotFoundError(f"FROST generated object not found: {object_path}")
        actual = _digest(object_path)
        if not isinstance(expected, str) or actual != expected:
            raise RuntimeError(f"FROST generated object digest mismatch: {object_path}")
        workspace_bytes = int(raw.get("workspace_bytes", -1))
        if workspace_bytes < 0 or workspace_bytes % 128:
            raise RuntimeError(
                f"{path}: kernel {artifact_id!r} workspace must be a nonnegative "
                "multiple of 128 bytes"
            )
        launch_tail = tuple(raw.get("launch", {}).get("tail", ()))
        if launch_tail not in (("output", "scale"), ("scale", "output")):
            raise RuntimeError(
                f"{path}: kernel {artifact_id!r} has unsupported launch tail "
                f"{launch_tail!r}"
            )
        result.append(
            FrostGroupedGemm1SwiGLUKernel(
                artifact_id=artifact_id,
                arch=str(raw.get("arch", "")),
                symbol=symbol,
                object_path=object_path,
                object_sha256=actual,
                workspace_bytes=workspace_bytes,
                contract=dict(raw.get("contract", {})),
                tactic_metadata=dict(raw.get("tactic", {})),
                launch_tail=launch_tail,
            )
        )
    return result


@functools.lru_cache(maxsize=8)
def _discover(roots: tuple[Path, ...]) -> tuple[FrostGroupedGemm1SwiGLUKernel, ...]:
    kernels = [kernel for root in roots for kernel in _read_root(root)]
    identities = [kernel.artifact_id for kernel in kernels]
    if len(identities) != len(set(identities)):
        raise RuntimeError("duplicate FROST kernel id across artifact roots")
    return tuple(kernels)


def clear_artifact_cache() -> None:
    _discover.cache_clear()
    _load_object.cache_clear()


def _arch_for(device: torch.device) -> str:
    major, minor = torch.cuda.get_device_capability(device)
    return f"sm_{major}{minor}a"


def _dimension_matches(value: int, rule: Any) -> bool:
    if isinstance(rule, int):
        return value == rule
    if not isinstance(rule, dict):
        return False
    multiple = int(rule.get("multiple_of", 1))
    if multiple <= 0:
        raise RuntimeError("FROST dimension 'multiple_of' must be positive")
    return (
        value >= int(rule.get("min", 0))
        and ("max" not in rule or value <= int(rule["max"]))
        and value % multiple == 0
    )


def _validate_common(
    grouped_tokens: torch.Tensor,
    gate_weights: torch.Tensor,
    up_weights: torch.Tensor,
    first_token_offset: torch.Tensor,
    scale: torch.Tensor,
    out: torch.Tensor | None,
) -> tuple[int, int, int, int, int]:
    if grouped_tokens.ndim != 2:
        raise ValueError("grouped_tokens must have shape [S, K]")
    if gate_weights.ndim != 3 or up_weights.shape != gate_weights.shape:
        raise ValueError("gate_weights and up_weights must share shape [E, N, K]")
    s, k = map(int, grouped_tokens.shape)
    e, n, weight_k = map(int, gate_weights.shape)
    if weight_k != k:
        raise ValueError(f"token K={k} does not match weight K={weight_k}")
    if first_token_offset.ndim != 1 or first_token_offset.dtype != torch.int32:
        raise ValueError("first_token_offset must be a one-dimensional int32 tensor")
    groups = int(first_token_offset.numel())
    if groups == 0:
        raise ValueError("first_token_offset must contain at least one group")
    if scale.dtype != torch.float32 or scale.numel() != 1:
        raise ValueError("scale must contain one float32 value")
    if out is not None and (tuple(out.shape) != (s, n) or out.dtype != torch.bfloat16):
        raise ValueError(f"out must be BF16 with shape {(s, n)}")
    tensors = [
        grouped_tokens,
        gate_weights,
        up_weights,
        first_token_offset,
        scale,
    ]
    if out is not None:
        tensors.append(out)
    if any(t.device != grouped_tokens.device for t in tensors):
        raise ValueError("all grouped GEMM1 tensors must be on the same device")
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("all grouped GEMM1 tensors must be contiguous")
    if any(t.dtype != torch.bfloat16 for t in tensors[:3]):
        raise ValueError("the v1 grouped GEMM1 ABI requires BF16 tokens and weights")
    return s, n, k, e, groups


def matching_kernels(
    grouped_tokens: torch.Tensor,
    gate_weights: torch.Tensor,
    up_weights: torch.Tensor,
    first_token_offset: torch.Tensor,
    scale: torch.Tensor,
    out: torch.Tensor | None = None,
) -> tuple[FrostGroupedGemm1SwiGLUKernel, ...]:
    s, n, k, e, groups = _validate_common(
        grouped_tokens,
        gate_weights,
        up_weights,
        first_token_offset,
        scale,
        out,
    )
    values = {"s": s, "n": n, "k": k, "experts": e, "groups": groups}
    return tuple(
        kernel
        for kernel in _discover(_artifact_roots())
        if kernel.arch == _arch_for(grouped_tokens.device)
        and all(
            _dimension_matches(values[name], kernel.contract.get(name))
            for name in values
        )
    )


def workspace_size(
    grouped_tokens: torch.Tensor,
    gate_weights: torch.Tensor,
    up_weights: torch.Tensor,
    first_token_offset: torch.Tensor,
    scale: torch.Tensor,
    out: torch.Tensor | None = None,
) -> int:
    kernels = matching_kernels(
        grouped_tokens,
        gate_weights,
        up_weights,
        first_token_offset,
        scale,
        out,
    )
    if not kernels:
        raise RuntimeError(
            "no generated FROST grouped GEMM1 + SwiGLU kernel matches this call"
        )
    return max(kernel.workspace_bytes for kernel in kernels)


@functools.lru_cache(maxsize=None)
def _load_object(path: Path, digest: str, symbol: str) -> Any:
    del digest  # Included in the cache key to make replacements a new module.
    from cutlass.cute.runtime import load_module

    module = load_module(str(path), enable_tvm_ffi=True)
    try:
        return getattr(module, symbol)
    except AttributeError:
        return module[symbol]


def _current_custream(device: torch.device) -> Any:
    from cuda.bindings import driver

    return driver.CUstream(torch.cuda.current_stream(device).cuda_stream)


def _launch(
    kernel: FrostGroupedGemm1SwiGLUKernel,
    grouped_tokens: torch.Tensor,
    gate_weights: torch.Tensor,
    up_weights: torch.Tensor,
    first_token_offset: torch.Tensor,
    scale: torch.Tensor,
    out: torch.Tensor,
    workspace: torch.Tensor,
) -> None:
    if workspace.dtype != torch.uint8 or not workspace.is_contiguous():
        raise ValueError("workspace must be a contiguous uint8 tensor")
    if workspace.device != grouped_tokens.device:
        raise ValueError("workspace must be on the grouped token device")
    if workspace.numel() < kernel.workspace_bytes:
        raise ValueError(
            f"kernel requires {kernel.workspace_bytes} workspace bytes; "
            f"got {workspace.numel()}"
        )
    if workspace.data_ptr() % 128:
        raise ValueError("FROST grouped GEMM workspace must be 128-byte aligned")

    # FROST's in-process CompiledMoeGemm wrapper resets the persistent grouped
    # scheduler counter before every launch.  The exported object deliberately
    # contains only the host/kernel launchable, so the embedding runtime must
    # perform that initialization.  Clearing the complete small descriptor
    # workspace is ABI-safe and also avoids depending on FROST at deployment.
    workspace[: kernel.workspace_bytes].zero_()
    launch = _load_object(kernel.object_path, kernel.object_sha256, kernel.symbol)
    token = grouped_tokens.unsqueeze(0).permute(1, 2, 0)
    gate = gate_weights.permute(1, 2, 0)
    up = up_weights.permute(1, 2, 0)
    output = out.unsqueeze(0).permute(1, 2, 0)
    s, k = map(int, grouped_tokens.shape)
    e, n, _ = map(int, gate_weights.shape)
    problem = (
        s,
        n,
        k,
        e,
        int(first_token_offset.numel()),
        *map(int, token.stride()),
        *map(int, gate.stride()),
        *map(int, up.stride()),
        *map(int, output.stride()),
    )
    workspace_i64 = workspace[: kernel.workspace_bytes].view(torch.int64)
    tail_tensors = {
        "output": output,
        "scale": scale.reshape(1, 1, 1),
    }
    launch(
        problem,
        first_token_offset,
        workspace_i64,
        token,
        gate,
        up,
        *(tail_tensors[name] for name in kernel.launch_tail),
        # Exported TVM-FFI functions do not accept keyword arguments, even
        # though the in-process CuTe callable uses ``stream=``.
        _current_custream(grouped_tokens.device),
    )


class FrostGroupedGemm1SwiGLURunner(TunableRunner):
    """One FlashInfer runner over all matching generated FROST tactics."""

    def get_valid_tactics(
        self, inputs: list[torch.Tensor], profile: OptimizationProfile
    ) -> list[Any]:
        del profile
        return [kernel.tactic for kernel in matching_kernels(*inputs[:6])]

    def _resolve(self, inputs: list[torch.Tensor], tactic: Any):
        kernels = matching_kernels(*inputs[:6])
        if not kernels:
            raise RuntimeError("no matching FROST grouped GEMM1 + SwiGLU artifact")
        if tactic == -1:
            return kernels[0]
        for kernel in kernels:
            if kernel.tactic == tactic:
                return kernel
        raise ValueError(f"unknown or stale FROST tactic: {tactic!r}")

    def validate_tactic(self, inputs: list[torch.Tensor], tactic: Any) -> bool:
        try:
            self._resolve(inputs, tactic)
            return True
        except (RuntimeError, ValueError):
            return False

    def forward(
        self,
        inputs: list[torch.Tensor],
        tactic: Any = -1,
        do_preparation: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        del do_preparation, kwargs
        kernel = self._resolve(inputs, tactic)
        _launch(kernel, *inputs)
        return inputs[5]


__all__ = [
    "FrostGroupedGemm1SwiGLUKernel",
    "FrostGroupedGemm1SwiGLURunner",
    "clear_artifact_cache",
    "matching_kernels",
    "workspace_size",
]
