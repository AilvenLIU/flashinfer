"""Full-MoE ablation: original CUTLASS/TRTLLM pool vs pool plus standalone Frost.

Times routing/permutation, FC1/SwiGLU, FC2 and finalize, using alternating CUDA
graph replay. Both original backends keep their unmodified tactic pools.
Frost-only timing is also reported, even when it loses cross-backend selection.
"""

import argparse
import json
import os
import statistics
import subprocess
from pathlib import Path

import torch


def check_idle_gpu(gpu_uuid):
    """Fail instead of reporting timings contaminated by another GPU process."""
    uuid = str(gpu_uuid).removeprefix("GPU-").lower()
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    peers = []
    for line in output.splitlines():
        device, pid = (value.strip() for value in line.split(",", 1))
        if device.removeprefix("GPU-").lower() == uuid and int(pid) != os.getpid():
            peers.append(int(pid))
    if peers:
        raise RuntimeError(
            f"Other compute processes on benchmark GPU: {peers}. "
            "Aborting: any timing from this interrupted run is not an isolated "
            "performance result. Retry when this GPU is idle; do not stop others' jobs."
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", default="8192,12288")
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--experts", type=int, default=12)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--intermediate", type=int, default=3072)
    parser.add_argument(
        "--probe-frost",
        action="store_true",
        help="Research-only: compare matching objects outside automatic admission",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help="Research-only object directory, used instead of packaged objects",
    )
    parser.add_argument(
        "--sweep-frost",
        action="store_true",
        help="Separately rank every full Frost tactic with rotated graph timing",
    )
    parser.add_argument("--routing", choices=("uniform", "skew"), default="uniform")
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--graph-batch", type=int, default=16)
    parser.add_argument("--source-jit", action="store_true")
    args = parser.parse_args()
    if min(args.rounds, args.iterations, args.graph_batch) <= 0:
        parser.error("rounds, iterations and graph-batch must be positive")
    gpu_uuid = torch.cuda.get_device_properties(torch.cuda.current_device()).uuid
    check_idle_gpu(gpu_uuid)
    if args.source_jit:
        from flashinfer.jit import env

        root = Path(__file__).resolve().parents[1]
        env.FLASHINFER_CSRC_DIR = root / "csrc"
        env.FLASHINFER_INCLUDE_DIR = root / "include"
        cccl = root / "3rdparty/cccl"
        env.CCCL_INCLUDE_DIRS = [
            cccl / "cub",
            cccl / "libcudacxx/include",
            cccl / "thrust",
        ]

    from flashinfer.autotuner import autotune
    from flashinfer.fused_moe import (
        BackendOptions,
        CutlassBf16Config,
        ExecutionConfig,
        ExpertConfig,
        MoEActivationPack,
        MoEConfig,
        MoELayer,
        MoEWeightPack,
        QuantConfig,
        RoutingConfig,
        TrtllmBf16Config,
    )
    from flashinfer.fused_moe.prepare import prepare_trtllm_bf16_weights

    torch.manual_seed(41)
    e, h, i = args.experts, args.hidden, args.intermediate
    if min(e, h, i, args.top_k) <= 0 or args.top_k > e:
        parser.error("positive dimensions and top-k <= experts are required")
    if args.artifact_root is not None:
        from flashinfer.experimental.frost_selected_kernels import fc2, runtime

        # Use one explicit pool, so re-testing an export after packaging it does
        # not produce duplicate artifact ids. Neither runtime module is changed
        # on disk, and normal application processes keep their packaged roots.
        roots = (args.artifact_root.resolve(),)
        runtime._artifact_roots = lambda: roots
        fc2._artifact_roots = lambda: roots
    tokens_list = [int(s) for s in args.tokens.split(",")]
    config = MoEConfig(
        routing=RoutingConfig(num_experts=e, top_k=args.top_k),
        quant=QuantConfig(),
        experts=ExpertConfig(intermediate_size=i),
        backend=BackendOptions((CutlassBf16Config(), TrtllmBf16Config())),
        execution=ExecutionConfig(tune_max_num_tokens=max(tokens_list)),
    )
    w1 = torch.randn(e, 2 * i, h, device="cuda", dtype=torch.bfloat16) * 0.02
    w2 = torch.randn(e, h, i, device="cuda", dtype=torch.bfloat16) * 0.02
    weights = MoEWeightPack(
        {
            "cutlass_bf16": dict(fc1_expert_weights=w1, fc2_expert_weights=w2),
            "trtllm_bf16_routed": prepare_trtllm_bf16_weights(
                w1,
                w2,
                num_local_experts=e,
                hidden_size=h,
                intermediate_size=i,
            ),
        }
    )
    print(
        json.dumps(
            dict(
                gpu=torch.cuda.get_device_name(),
                experts=e,
                hidden=h,
                intermediate=i,
                top_k=args.top_k,
                routing=args.routing,
                mode="cuda_graph",
                torch=torch.__version__,
                rounds=args.rounds,
                iterations=args.iterations,
                graph_batch=args.graph_batch,
                probe_frost=args.probe_frost,
                artifact_root=str(args.artifact_root) if args.artifact_root else None,
            )
        ),
        flush=True,
    )
    for tokens in tokens_list:
        x = torch.randn(tokens, h, device="cuda", dtype=torch.bfloat16) * 0.02
        logits = torch.rand(tokens, e, device="cuda")
        if args.routing == "skew":
            logits[: tokens // 2, 0] += 2
        ids = logits.topk(args.top_k, dim=1).indices.int()
        scores = torch.rand(tokens, args.top_k, device="cuda").softmax(-1)
        act = MoEActivationPack(x, None, ids, scores)
        layers = {"original": MoELayer(config), "with_frost": MoELayer(config)}
        # Benchmark-only dispatcher ablation, never a change to an existing runner.
        layers["original"]._additional_frost_candidate = lambda *args: None
        if args.probe_frost:
            from flashinfer.experimental.frost_selected_kernels.moe import (
                FrostBf16MoeRunner,
            )

            probe = FrostBf16MoeRunner(config, x.device)
            probe.check_support()
            probe.build()
            layers["with_frost"]._frost_runner = probe
            layers["with_frost"]._additional_frost_candidate = lambda *args: probe
        outputs, graphs, winners = {}, {}, {}
        for label, layer in layers.items():
            with autotune():
                layer(act, weights)
            check_idle_gpu(gpu_uuid)
            outputs[label] = layer(act, weights).clone()
            winners[label] = [(r.backend_key, t) for r, t in layer._winners.values()]
            for _ in range(5):
                layer(act, weights)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(args.graph_batch):
                    layer(act, weights)
            graphs[label] = graph
            print(
                json.dumps(
                    dict(tokens=tokens, pool=label, winners=winners[label]),
                    default=list,  # TRTLLM may return a TVM-FFI Array tactic.
                ),
                flush=True,
            )

        # Measure the best independent Frost candidate even if an original backend wins.
        layer = layers["with_frost"]
        frost = layer._frost_runner
        if frost is not None and (args.probe_frost or frost.accepts(act, weights)):
            packed = frost.pack_inputs(act, weights)
            _, tactic = layer.tuner.choose_one(
                custom_op="moe_frost_bf16",
                runners=[frost],
                inputs=packed,
                tuning_config=frost.tuning_config_for(packed),
                **frost.launch_kwargs_for(packed),
            )
            outputs["frost_only"] = frost.forward(packed, tactic).clone()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(args.graph_batch):
                    frost.forward(packed, tactic)
            graphs["frost_only"] = graph
            print(
                json.dumps(
                    dict(
                        tokens=tokens,
                        pool="frost_only",
                        tactic=tactic,
                        candidate_count=len(frost.get_valid_tactics(packed, None)),
                        workspace_bytes=packed.launch_state.workspace.numel(),
                    )
                ),
                flush=True,
            )
        errors = {
            label: (
                (out.float() - outputs["original"].float()).norm()
                / outputs["original"].float().norm()
            ).item()
            for label, out in outputs.items()
            if label != "original"
        }
        assert all(error < 0.015 for error in errors.values()), errors
        samples = {label: [] for label in graphs}
        for round_idx in range(args.rounds):
            check_idle_gpu(gpu_uuid)
            labels = list(graphs)
            offset = round_idx % len(labels)
            labels = labels[offset:] + labels[:offset]
            if round_idx % 2:
                labels.reverse()
            for label in labels:
                graphs[label].replay()
                start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                start.record()
                for _ in range(args.iterations):
                    graphs[label].replay()
                end.record()
                end.synchronize()
                samples[label].append(
                    start.elapsed_time(end) / (args.iterations * args.graph_batch)
                )
        check_idle_gpu(gpu_uuid)
        medians = {label: statistics.median(times) for label, times in samples.items()}
        print(
            json.dumps(
                dict(
                    tokens=tokens,
                    routed_rows=tokens * args.top_k,
                    milliseconds=medians,
                    speedup=medians["original"] / medians["with_frost"],
                    relative_l2=errors,
                    samples_ms=samples,
                )
            ),
            flush=True,
        )
        if args.sweep_frost and frost is not None:
            packed = frost.pack_inputs(act, weights)
            tactics = frost.get_valid_tactics(packed, None)
            sweep_graphs = []
            for tactic in tactics:
                actual = frost.forward(packed, tactic)
                error = (
                    (actual.float() - outputs["original"].float()).norm()
                    / outputs["original"].float().norm()
                ).item()
                if error >= 0.015 or not torch.isfinite(actual).all().item():
                    raise RuntimeError(
                        f"Frost tactic failed correctness: {tactic}: {error}"
                    )
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(8):
                        frost.forward(packed, tactic)
                actual.fill_(float("nan"))
                graph.replay()
                error = (
                    (actual.float() - outputs["original"].float()).norm()
                    / outputs["original"].float().norm()
                ).item()
                if not error < 0.015:
                    raise RuntimeError(
                        f"Frost graph tactic failed correctness: {tactic}: {error}"
                    )
                sweep_graphs.append(graph)
            times = [[] for _ in tactics]
            for round_idx in range(5):
                check_idle_gpu(gpu_uuid)
                order = list(range(len(tactics)))
                shift = (round_idx * 3) % len(order)
                order = order[shift:] + order[:shift]
                if round_idx % 2:
                    order.reverse()
                for idx in order:
                    graph = sweep_graphs[idx]
                    graph.replay()
                    start, end = (
                        torch.cuda.Event(enable_timing=True) for _ in range(2)
                    )
                    start.record()
                    for _ in range(5):
                        graph.replay()
                    end.record()
                    end.synchronize()
                    times[idx].append(start.elapsed_time(end) / 40)
            check_idle_gpu(gpu_uuid)
            ranked = sorted(
                (
                    dict(milliseconds=statistics.median(t), tactic=tactic)
                    for t, tactic in zip(times, tactics, strict=True)
                ),
                key=lambda x: x["milliseconds"],
            )
            print(json.dumps(dict(tokens=tokens, frost_sweep=ranked)), flush=True)


if __name__ == "__main__":
    main()
