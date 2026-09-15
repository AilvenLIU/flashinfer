"""Full independent Frost MoE, including routing, both GEMMs and finalize."""

import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from flashinfer.experimental.frost_selected_kernels import moe
from flashinfer.fused_moe import (
    BackendOptions,
    CutlassBf16Config,
    ExecutionConfig,
    ExpertConfig,
    MoEActivationPack,
    MoEConfig,
    MoEFinalizeConfig,
    MoELayer,
    MoEWeightPack,
    QuantConfig,
    RoutingConfig,
    SwiGLU,
)

sm100 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
    reason="packaged Frost MoE objects target SM100a",
)


def config(topk=2, experts=8, intermediate=256, ceiling=16384):
    return MoEConfig(
        routing=RoutingConfig(num_experts=experts, top_k=topk),
        quant=QuantConfig(),
        experts=ExpertConfig(intermediate_size=intermediate),
        backend=BackendOptions((CutlassBf16Config(),)),
        execution=ExecutionConfig(enable_pdl=False, tune_max_num_tokens=ceiling),
    )


def packs(tokens=129, topk=2, experts=8, hidden=128, intermediate=256, device="cuda"):
    x = torch.randn(tokens, hidden, dtype=torch.bfloat16, device=device) * 0.1
    w1 = (
        torch.randn(experts, 2 * intermediate, hidden, dtype=x.dtype, device=device)
        * 0.1
    )
    w2 = torch.randn(experts, hidden, intermediate, dtype=x.dtype, device=device) * 0.1
    ids = torch.randint(
        0, experts // 2, (tokens, topk), dtype=torch.int32, device=device
    )
    scores = torch.rand(tokens, topk, dtype=torch.float32, device=device)
    scores /= scores.sum(dim=1, keepdim=True)
    act = MoEActivationPack(x, None, ids, scores)
    weights = MoEWeightPack(
        {"cutlass_bf16": dict(fc1_expert_weights=w1, fc2_expert_weights=w2)}
    )
    return act, weights


def reference(act, weights):
    x, ids, scores = act.hidden_states_q, act.topk_ids, act.topk_weights
    w = weights.get_view("cutlass_bf16")
    w1, w2 = w["fc1_expert_weights"], w["fc2_expert_weights"]
    i = w2.shape[-1]
    expanded = torch.zeros(*ids.shape, x.shape[1], device=x.device, dtype=torch.float32)
    for expert in range(w1.shape[0]):
        token, slot = torch.where(ids == expert)
        up = x[token].float() @ w1[expert, :i].float().T
        gate = x[token].float() @ w1[expert, i:].float().T
        mid = (F.silu(gate) * up).bfloat16()
        down = (mid.float() @ w2[expert].float().T).bfloat16()
        expanded[token, slot] = down.float() * scores[token, slot, None]
    return expanded.sum(dim=1).bfloat16()


@sm100
@pytest.mark.parametrize("topk", [1, 2, 4])
def test_all_compound_tactics_full_moe_graph_and_dynamic_routing(topk, monkeypatch):
    # Neither the Frost compiler nor the CUTLASS execution API is needed.
    monkeypatch.setitem(sys.modules, "cudnn.gemm.frost.compiler", None)
    import flashinfer.fused_moe as fused

    def forbidden(*args, **kwargs):
        raise AssertionError("independent Frost must not invoke CUTLASS")

    monkeypatch.setattr(fused, "cutlass_fused_moe", forbidden)
    torch.manual_seed(59)
    act, weights = packs(topk=topk)
    runner = moe.FrostBf16MoeRunner(config(topk), "cuda")
    runner.check_support()
    runner.build()
    inputs = runner.pack_inputs(act, weights)
    expected = reference(act, weights)
    tactics = runner.get_valid_tactics(inputs, None)
    assert len(tactics) == 16
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for tactic in tactics:
            out = runner.forward(inputs, tactic)
            torch.testing.assert_close(out, expected, atol=2e-4, rtol=2e-2)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            runner.forward(inputs, tactics[-1])
        # Change contents in-place: captured routing cannot depend on host counts.
        act.topk_ids.fill_(7)
        act.topk_weights.mul_(0.5)
        act.hidden_states_q.mul_(0.5)
        expected = reference(act, weights)
        for _ in range(3):
            out.fill_(float("nan"))
            graph.replay()
        torch.testing.assert_close(out, expected, atol=2e-4, rtol=2e-2)
        # Invalid ids must not escape the workspace or become weight addresses.
        act.topk_ids[::2] = -1
        act.topk_ids[1::4] = 8
        graph.replay()
        torch.testing.assert_close(out, reference(act, weights), atol=2e-4, rtol=2e-2)
    torch.cuda.current_stream().wait_stream(stream)


@sm100
def test_interleaved_packs_do_not_exchange_weights_and_reject_stale_tactics():
    runner = moe.FrostBf16MoeRunner(config(), "cuda")
    runner.check_support()
    runner.build()
    a, wa = packs(tokens=17)
    b, wb = packs(tokens=33)
    pb = runner.pack_inputs(b, wb)
    pa = runner.pack_inputs(a, wa)
    assert pa.tuning_config.cuda_graph_profile_replays == 3
    assert pa.launch_state.workspace.data_ptr() == pb.launch_state.workspace.data_ptr()
    assert len(runner._workspace_pool) == 1
    # Simulate the plain-list inputs synthesized by the autotuner.
    runner.forward(list(pa), **runner.launch_kwargs_for(pa))
    runner.forward(pb)
    torch.testing.assert_close(pa[0], reference(a, wa), atol=2e-4, rtol=2e-2)
    torch.testing.assert_close(pb[0], reference(b, wb), atol=2e-4, rtol=2e-2)
    with pytest.raises(ValueError, match="stale"):
        runner.forward(pa, ("missing",))
    with pytest.raises(ValueError, match="launch_state"):
        runner.forward(list(pa))
    wa.native_views["cutlass_bf16"]["gemm1_alpha"] = torch.ones(8, device="cuda")
    assert not runner.accepts(a, wa)
    with pytest.raises(ValueError, match="overrides"):
        runner.pack_inputs(a, wa)


@sm100
@pytest.mark.parametrize(
    "experts,hidden,intermediate",
    [(12, 7168, 3072), (8, 4096, 14336)],
)
def test_model_geometry_artifacts_ragged_graph(
    experts, hidden, intermediate, monkeypatch
):
    """Exercise every packaged pair on partial tiles and initially empty experts."""
    monkeypatch.setitem(sys.modules, "cudnn.gemm.frost.compiler", None)
    torch.manual_seed(97)
    act, weights = packs(
        tokens=129, experts=experts, hidden=hidden, intermediate=intermediate
    )
    runner = moe.FrostBf16MoeRunner(
        config(experts=experts, intermediate=intermediate), "cuda"
    )
    runner.check_support()
    runner.build()
    packed = runner.pack_inputs(act, weights)
    expected = reference(act, weights)
    expected_norm = expected.float().norm()
    tactics = runner.get_valid_tactics(packed, None)
    assert tactics
    for tactic in tactics:
        actual = runner.forward(packed, tactic)
        assert torch.isfinite(actual).all().item()
        error = (actual.float() - expected.float()).norm() / expected_norm
        assert error.item() < 0.01, (tactic, error.item())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            runner.forward(packed, tactic)
        actual.fill_(float("nan"))
        graph.replay()
        error = (actual.float() - expected.float()).norm() / expected_norm
        assert error.item() < 0.01, (tactic, error.item())
    # Captured routing must work when previously empty experts become populated.
    act.topk_ids.fill_(experts - 1)
    act.topk_weights.mul_(0.5)
    graph.replay()
    expected = reference(act, weights)
    error = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert error.item() < 0.01


@sm100
@pytest.mark.parametrize(
    "overrides",
    [
        {"activation": SwiGLU(alpha=2.0)},
        {"finalize": MoEFinalizeConfig(do_finalize=False)},
        {"experts": ExpertConfig(intermediate_size=256, local_expert_offset=1)},
    ],
)
def test_unsupported_semantics_are_not_automatic_candidates(overrides):
    assert moe.automatic_candidate(replace(config(), **overrides), "cuda") is None


def test_layer_adds_independent_candidate_and_separates_winner_cache(monkeypatch):
    # Dispatch-policy test without allocating multi-GB weights or benchmarking.
    from flashinfer.fused_moe import layer as layer_module

    cfg = config(experts=12, intermediate=3072)
    calls = []

    class FakeRunner:
        supported_routing_modes = moe.FrostBf16MoeRunner.supported_routing_modes

        def __init__(self, key):
            self.backend_key = key

        def accepts(self, act, weights):
            return not weights.native_views.get("override")

        def pack_inputs(self, act, weights):
            return [act.hidden_states_q]

        def launch_kwargs_for(self, inputs):
            return {}

        def forward(self, inputs, **kwargs):
            return inputs[0]

    old, frost = FakeRunner("cutlass_bf16"), FakeRunner("frost_bf16")
    layer = MoELayer.__new__(MoELayer)
    layer.config, layer.device, layer._arch = cfg, torch.device("cuda", 0), 100
    layer.tuner = SimpleNamespace(is_tuning_mode=True)
    layer.runners, layer._frost_runner, layer._winners = [old], None, {}
    monkeypatch.setattr(moe, "automatic_candidate", lambda *args: frost)

    def select(act, weights, runners):
        calls.append([r.backend_key for r in runners])
        return runners[-1], -1

    monkeypatch.setattr(layer, "_select_winner", select)
    act, weights = packs(
        tokens=4096, experts=12, hidden=7168, intermediate=3072, device="meta"
    )
    layer(act, weights)
    assert layer.winner_backend == "frost_bf16"
    assert calls == [["cutlass_bf16", "frost_bf16"]]
    assert layer.runners == [old]  # no mutation of original backend collection
    layer.tuner.is_tuning_mode = False
    layer(act, weights)
    assert len(calls) == 1
    weights.native_views["override"] = {"present": True}
    layer(act, weights)
    assert layer.winner_backend == "cutlass_bf16"
    weights.native_views.pop("override")
    # Same old tuning bucket, but another exact token count needs its own plan.
    monkeypatch.setattr(layer_module, "map_to_hybrid_bucket", lambda *args: 4096)
    act2, weights2 = packs(
        tokens=4097, experts=12, hidden=7168, intermediate=3072, device="meta"
    )
    layer(act2, weights2)
    assert calls[-1] == ["cutlass_bf16", "frost_bf16"]
    assert len(calls) == 3
    layer.reset_winner()
    assert not layer._winners


@sm100
@pytest.mark.parametrize(
    "experts,hidden,intermediate",
    [(12, 7168, 3072), (8, 4096, 14336)],
)
def test_original_layer_api_can_execute_winning_frost_and_replay(
    experts, hidden, intermediate, monkeypatch
):
    from flashinfer.autotuner import autotune

    torch.manual_seed(83)
    act, weights = packs(
        tokens=4096, experts=experts, hidden=hidden, intermediate=intermediate
    )
    layer = MoELayer(config(experts=experts, intermediate=intermediate))
    visited = []

    def select(act, weights, runners):
        # Force a winner to test dispatch independently of machine performance.
        visited.extend(r.backend_key for r in runners)
        frost = next(r for r in runners if r.backend_key == "frost_bf16")
        packed = frost.pack_inputs(act, weights)
        tactic = frost.get_valid_tactics(packed, None)[0]
        frost.forward(packed, tactic)  # warm native objects before capture
        return frost, tactic

    monkeypatch.setattr(layer, "_select_winner", select)
    with autotune():
        actual = layer(act, weights)
    assert visited == ["cutlass_bf16", "frost_bf16"]
    assert layer.winner_backend == "frost_bf16"
    expected = reference(act, weights)
    rel_l2 = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert rel_l2.item() < 0.01
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = layer(act, weights)
    act.topk_ids.fill_(experts - 1)
    act.topk_weights.mul_(0.5)
    graph.replay()
    expected = reference(act, weights)
    rel_l2 = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert rel_l2.item() < 0.01
    assert len(visited) == 2  # cached winning runner, no re-tune on replay


def test_auto_admission_only_large_supported_geometry():
    from flashinfer.experimental.frost_selected_kernels.support import large_bf16_moe

    cfg = config(experts=12, intermediate=3072)
    for tokens, hidden, arch, expected in (
        (4095, 7168, 100, False),
        (4096, 7168, 100, True),
        (4096, 7168, 103, False),
        (4096, 4096, 100, False),
    ):
        act, _ = packs(
            tokens=tokens, experts=12, hidden=hidden, intermediate=3072, device="meta"
        )
        assert large_bf16_moe(cfg, act, arch) == expected
    cfg = config(experts=8, intermediate=14336)
    for tokens, expected in [(4095, False), (4096, True)]:
        act, _ = packs(
            tokens=tokens, experts=8, hidden=4096, intermediate=14336, device="meta"
        )
        assert large_bf16_moe(cfg, act, 100) == expected
    cfg = config(topk=6, experts=64, intermediate=1408)
    act, _ = packs(
        tokens=4096,
        topk=6,
        experts=64,
        hidden=2048,
        intermediate=1408,
        device="meta",
    )
    assert not large_bf16_moe(cfg, act, 100)  # research-only, not auto-admitted


def test_support_import_does_not_load_execution_implementation():
    import subprocess

    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import flashinfer.experimental.frost_selected_kernels.support; "
            "assert 'flashinfer.experimental.frost_selected_kernels.runtime' not in sys.modules; "
            "assert 'flashinfer.experimental.frost_selected_kernels.moe' not in sys.modules",
        ],
        check=True,
    )


def test_benchmark_rejects_concurrent_same_gpu_processes(monkeypatch):
    import os
    import runpy
    import subprocess
    from pathlib import Path

    benchmark = Path(__file__).resolve().parents[2] / "benchmarks" / "frost_bf16_moe.py"
    check = runpy.run_path(str(benchmark))["check_idle_gpu"]
    mine = f"GPU-test, {os.getpid()}\n"
    monkeypatch.setattr(
        subprocess, "check_output", lambda *a, **kw: mine + "GPU-other, 999\n"
    )
    check("test")
    monkeypatch.setattr(
        subprocess, "check_output", lambda *a, **kw: mine + "GPU-test, 999\n"
    )
    with pytest.raises(RuntimeError, match="Other compute processes"):
        check("test")
