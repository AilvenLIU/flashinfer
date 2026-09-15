"""Manifest and ABI tests for FROST grouped GEMM1 + SwiGLU artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from flashinfer.experimental.frost_selected_kernels import runtime


def _install_artifact(tmp_path, monkeypatch, *, digest: str | None = None):
    obj_dir = tmp_path / "objects"
    obj_dir.mkdir()
    obj = obj_dir / "grouped_swiglu.o"
    obj.write_bytes(b"generated-frost-grouped-swiglu-object")
    actual = hashlib.sha256(obj.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "producer": {"name": "frost"},
        "kernels": [
            {
                "id": "grouped-swiglu-sm100-tile-a",
                "op": "grouped_gemm1_swiglu",
                "arch": "sm_100a",
                "abi": "frost_grouped_gemm1_swiglu_v1",
                "symbol": "flashinfer_frost_grouped_swiglu",
                "object": {
                    "path": "objects/grouped_swiglu.o",
                    "sha256": actual if digest is None else digest,
                },
                "workspace_bytes": 256,
                "launch": {"tail": ["scale", "output"]},
                "contract": {
                    "s": {"min": 1},
                    "n": 24,
                    "k": 16,
                    "experts": 3,
                    "groups": 3,
                    "activation": "silu(gate) * up",
                },
                "tactic": {
                    "template": "sm100_moe_grouped_matmul_fwd_2ctamma.py",
                    "tile": "tile-a",
                    "cta_group": 2,
                    "scheduler": "clc",
                },
                "producer_revision": "test-revision",
            }
        ],
    }
    (tmp_path / "frost_selected_kernels.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(runtime, "_artifact_roots", lambda: (tmp_path,))
    runtime.clear_artifact_cache()
    monkeypatch.setattr(runtime, "_arch_for", lambda _device: "sm_100a")
    return obj


def _inputs():
    grouped_tokens = torch.empty((7, 16), dtype=torch.bfloat16)
    gate = torch.empty((3, 24, 16), dtype=torch.bfloat16)
    up = torch.empty_like(gate)
    offsets = torch.tensor([0, 2, 5], dtype=torch.int32)
    scale = torch.tensor([0.5], dtype=torch.float32)
    out = torch.empty((7, 24), dtype=torch.bfloat16)
    storage = torch.empty(256 + 127, dtype=torch.uint8)
    shift = (-storage.data_ptr()) % 128
    workspace = storage[shift : shift + 256]
    return [grouped_tokens, gate, up, offsets, scale, out, workspace]


def test_manifest_provides_stable_tactic_and_workspace(tmp_path, monkeypatch):
    _install_artifact(tmp_path, monkeypatch)
    inputs = _inputs()
    kernels = runtime.matching_kernels(*inputs[:6])
    assert len(kernels) == 1
    assert kernels[0].tactic == (
        "frost-grouped-swiglu-v1",
        "grouped-swiglu-sm100-tile-a",
        kernels[0].object_sha256[:20],
    )
    assert runtime.workspace_size(*inputs[:5]) == 256


def test_runtime_only_searches_packaged_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("FLASHINFER_FROST_KERNEL_PATH", str(tmp_path))
    packaged = Path(runtime.__file__).resolve().parent / "artifacts"
    assert runtime._artifact_roots() == (packaged,)


def test_packaged_swiglu_has_multiple_tactics_and_prefers_n128_first():
    packaged = Path(runtime.__file__).resolve().parent / "artifacts"
    manifest = json.loads((packaged / "frost_selected_kernels.json").read_text())
    kernels = [
        item for item in manifest["kernels"] if item["op"] == "grouped_gemm1_swiglu"
    ]
    assert len(kernels) > 1
    assert kernels[0]["tactic"]["cta_tile"]["n"] == 128
    assert len({item["id"] for item in kernels}) == len(kernels)
    typical = [
        item
        for item in kernels
        if item["contract"]["n"] == 3072 and item["contract"]["k"] == 7168
    ]
    assert {
        (item["tactic"]["cta_tile"]["n"], item["tactic"]["store_mode"])
        for item in typical
    } == {(128, "stg"), (128, "tma"), (256, "stg"), (256, "tma")}


def test_manifest_rejects_modified_object(tmp_path, monkeypatch):
    _install_artifact(tmp_path, monkeypatch, digest="0" * 64)
    with pytest.raises(RuntimeError, match="digest mismatch"):
        runtime.matching_kernels(*_inputs()[:6])


def test_grouped_swiglu_runner_marshals_frost_abi(tmp_path, monkeypatch):
    _install_artifact(tmp_path, monkeypatch)
    calls = []

    def launch(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(runtime, "_load_object", lambda *_args: launch)
    monkeypatch.setattr(runtime, "_current_custream", lambda _device: "stream")
    inputs = _inputs()
    runner = runtime.FrostGroupedGemm1SwiGLURunner()
    tactic = runner.get_valid_tactics(inputs, profile=None)[0]
    assert runner.forward(inputs, tactic=tactic) is inputs[5]

    args, kwargs = calls[0]
    assert args[0][:5] == (7, 24, 16, 3, 3)
    assert tuple(args[3].shape) == (7, 16, 1)
    assert tuple(args[4].shape) == (24, 16, 3)
    assert tuple(args[5].shape) == (24, 16, 3)
    assert tuple(args[6].shape) == (1, 1, 1)
    assert tuple(args[7].shape) == (7, 24, 1)
    assert args[8] == "stream"
    assert args[2].dtype == torch.int64 and args[2].numel() == 32
    assert kwargs == {}


def test_contract_rejects_wrong_expert_shape(tmp_path, monkeypatch):
    _install_artifact(tmp_path, monkeypatch)
    inputs = _inputs()
    inputs[1] = inputs[1][:2]
    inputs[2] = inputs[2][:2]
    assert not runtime.matching_kernels(*inputs[:6])


def test_public_entry_points_are_experimental():
    from flashinfer.fused_moe import (
        frost_grouped_gemm1_swiglu,
        frost_grouped_gemm1_swiglu_workspace_size,
    )

    assert frost_grouped_gemm1_swiglu.is_experimental
    assert frost_grouped_gemm1_swiglu_workspace_size.is_experimental
