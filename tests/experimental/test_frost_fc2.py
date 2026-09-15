"""Offline BF16 FC2 artifacts: integrity, ragged groups, reuse and CUDA graphs."""

import json
import sys
from dataclasses import replace

import pytest
import torch

from flashinfer.experimental.frost_selected_kernels import fc2, runtime


def test_fc2_manifest_is_separate_from_fc1():
    kernels = fc2.discover()
    assert kernels
    assert all(k.contract["activation"] == "identity" for k in kernels)
    assert not {k.artifact_id for k in kernels} & {
        k.artifact_id for k in runtime._discover(runtime._artifact_roots())
    }
    assert len({k.tactic for k in kernels}) == len(kernels)


def test_fc2_manifest_rejects_modified_object(tmp_path):
    # Use an existing verified entry with a deliberately invalid digest.
    root = runtime._artifact_roots()[0]
    payload = json.loads((root / "frost_selected_kernels.json").read_text())
    entry = next(k for k in payload["kernels"] if k["op"] == "grouped_gemm2")
    entry["object"] = {"path": "bad.o", "sha256": "0" * 64}
    (tmp_path / "bad.o").write_bytes(b"bad object")
    (tmp_path / "frost_selected_kernels.json").write_text(
        json.dumps(dict(schema_version=1, kernels=[entry]))
    )
    with pytest.raises(RuntimeError, match="digest mismatch"):
        fc2.discover((tmp_path,))


sm100 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
    reason="packaged Frost FC2 objects target SM100a",
)


def reference(x, weights, starts):
    out = torch.empty(
        x.shape[0], weights.shape[1], dtype=torch.bfloat16, device=x.device
    )
    for expert, (begin, end) in enumerate(
        zip(starts, starts[1:] + [x.shape[0]], strict=True)
    ):
        out[begin:end] = (x[begin:end].float() @ weights[expert].float().T).bfloat16()
    return out


@sm100
@pytest.mark.parametrize("rows", [24, 513, 8192])
def test_fc2_all_configs_ragged_empty_graph(rows, monkeypatch):
    monkeypatch.setitem(sys.modules, "cudnn.gemm.frost.compiler", None)
    torch.manual_seed(63)
    e, h, i = 8, 128, 256
    x = torch.randn(rows, i, dtype=torch.bfloat16, device="cuda") * 0.1
    w = torch.randn(e, h, i, dtype=torch.bfloat16, device="cuda") * 0.1
    starts = [0, 0, 1, rows // 4, rows // 4, rows // 2, rows - 1, rows]
    offsets = torch.tensor(starts, dtype=torch.int32, device="cuda")
    out = torch.empty(rows, h, dtype=torch.bfloat16, device="cuda")
    expected = reference(x, w, starts)
    kernels = fc2.matching_kernels(rows, h, i, e, x.device)
    assert kernels
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for kernel in kernels:
            plan = fc2.PreparedFc2(kernel, x, w, offsets, out)
            for _ in range(3):
                assert plan.run() is out
            torch.testing.assert_close(out, expected, atol=2e-3, rtol=1e-2)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                plan.run()
            for _ in range(3):
                out.fill_(float("nan"))
                graph.replay()
            torch.testing.assert_close(out, expected, atol=2e-3, rtol=1e-2)
            # Re-bind contents (not pointers) without a compiler or new plan.
            x.mul_(0.5)
            plan.run()
            expected = reference(x, w, starts)
            torch.testing.assert_close(out, expected, atol=2e-3, rtol=1e-2)
    torch.cuda.current_stream().wait_stream(stream)


@sm100
def test_fc2_rejects_invalid_contract():
    x = torch.empty(32, 256, dtype=torch.bfloat16, device="cuda")
    w = torch.empty(8, 128, 256, dtype=torch.bfloat16, device="cuda")
    offsets = torch.arange(8, dtype=torch.int32, device="cuda") * 4
    out = torch.empty(32, 128, dtype=torch.bfloat16, device="cuda")
    kernel = fc2.matching_kernels(32, 128, 256, 8, x.device)[0]
    small = torch.empty(kernel.workspace_bytes - 1, dtype=torch.uint8, device="cuda")
    with pytest.raises(ValueError, match="workspace"):
        fc2.PreparedFc2(kernel, x, w, offsets, out, small)
    with pytest.raises(ValueError, match="geometry"):
        fc2.PreparedFc2(replace(kernel, contract={}), x, w, offsets, out)
    with pytest.raises(ValueError, match="int32"):
        fc2.PreparedFc2(kernel, x, w, offsets.long(), out)
    offsets[0] = 1
    with pytest.raises(ValueError, match="offsets"):
        fc2.PreparedFc2(kernel, x, w, offsets, out)


@sm100
def test_fc1_fc2_chain_without_cutlass_or_frost_compiler(monkeypatch):
    monkeypatch.setitem(sys.modules, "cudnn.gemm.frost.compiler", None)
    torch.manual_seed(79)
    s, e, h, i = 513, 8, 128, 256
    x = torch.randn(s, h, dtype=torch.bfloat16, device="cuda") * 0.1
    gate = torch.randn(e, i, h, dtype=torch.bfloat16, device="cuda") * 0.1
    up = torch.randn_like(gate) * 0.1
    down = torch.randn(e, h, i, dtype=torch.bfloat16, device="cuda") * 0.1
    starts = [0, 0, 1, 64, 64, 256, 512, 513]
    offsets = torch.tensor(starts, dtype=torch.int32, device="cuda")
    scale = torch.ones(1, dtype=torch.float32, device="cuda")
    intermediate = torch.empty(s, i, dtype=torch.bfloat16, device="cuda")
    out = torch.empty_like(x)
    first = runtime.matching_kernels(x, gate, up, offsets, scale, intermediate)[0]
    scratch = torch.empty(first.workspace_bytes, dtype=torch.uint8, device="cuda")
    second = fc2.matching_kernels(s, h, i, e, x.device)[0]
    plan = fc2.PreparedFc2(second, intermediate, down, offsets, out)

    def run():
        runtime._launch(first, x, gate, up, offsets, scale, intermediate, scratch)
        plan.run()

    expected = torch.empty_like(x)
    for j, (a, b) in enumerate(zip(starts, starts[1:] + [s], strict=True)):
        hidden = (
            torch.nn.functional.silu(x[a:b].float() @ gate[j].float().T)
            * (x[a:b].float() @ up[j].float().T)
        ).bfloat16()
        expected[a:b] = (hidden.float() @ down[j].float().T).bfloat16()
    run()
    torch.testing.assert_close(out, expected, atol=2e-4, rtol=2e-2)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(3):
        out.fill_(float("nan"))
        graph.replay()
    torch.testing.assert_close(out, expected, atol=2e-4, rtol=2e-2)
