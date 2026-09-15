"""Compare packaged FROST GEMM1+SwiGLU with an unfused BF16 baseline.

This benchmark starts after MoE routing/permutation and ends at the SwiGLU
output.  The baseline is one strided-batched GEMM with a concatenated 2N
weight followed by FP32 SiLU/multiply and a BF16 store.  It is intentionally
limited to uniform expert loads, for which strided-batched GEMM represents the
grouped operation without introducing a Python loop over experts.
"""

from __future__ import annotations

import argparse
import statistics
from collections.abc import Callable

import torch

import flashinfer
from flashinfer.experimental.frost_selected_kernels.runtime import workspace_size


def _csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(","))


def _one_sample(fn: Callable[[], None], iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def _interleaved_medians(
    candidates: dict[str, Callable[[], None]], *, rounds: int, iterations: int
) -> dict[str, float]:
    labels = list(candidates)
    samples: dict[str, list[float]] = {label: [] for label in labels}
    for round_index in range(rounds):
        offset = round_index % len(labels)
        order = labels[offset:] + labels[:offset]
        if round_index % 2:
            order.reverse()
        for label in order:
            samples[label].append(_one_sample(candidates[label], iterations))
    return {label: statistics.median(values) for label, values in samples.items()}


def _capture(fn: Callable[[], None]) -> torch.cuda.CUDAGraph:
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experts", type=int, default=12)
    parser.add_argument("--n", type=int, default=3072)
    parser.add_argument("--k", type=int, default=7168)
    parser.add_argument(
        "--tokens", type=_csv_ints, default=_csv_ints("24,96,384,1536,6144,12288")
    )
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--cuda-graph", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(41)
    device = torch.device("cuda")
    gate = (
        torch.randn(args.experts, args.n, args.k, device=device, dtype=torch.bfloat16)
        * 0.02
    ).contiguous()
    up = (torch.randn_like(gate) * 0.02).contiguous()
    # The conventional GEMM1 layout produces gate and up in one 2N GEMM.
    combined = torch.cat((gate, up), dim=1).contiguous()
    combined_t = combined.transpose(1, 2)
    scale = torch.ones(1, device=device, dtype=torch.float32)

    mode = "CUDA Graph replay" if args.cuda_graph else "eager public API"
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"mode: {mode}")
    print("baseline: 1x BF16 torch.bmm(2N) + FP32 SiLU/mul + BF16 store")
    for tokens in args.tokens:
        if tokens % args.experts:
            raise ValueError("uniform baseline requires tokens divisible by experts")
        group_m = tokens // args.experts
        x = (
            torch.randn(tokens, args.k, device=device, dtype=torch.bfloat16) * 0.02
        ).contiguous()
        offsets = torch.arange(args.experts, device=device, dtype=torch.int32) * group_m
        frost_out = torch.empty(tokens, args.n, device=device, dtype=torch.bfloat16)
        baseline_out = torch.empty(
            args.experts, group_m, args.n, device=device, dtype=torch.bfloat16
        )
        workspace = torch.empty(
            workspace_size(x, gate, up, offsets, scale, frost_out),
            device=device,
            dtype=torch.uint8,
        )
        x_grouped = x.view(args.experts, group_m, args.k)

        def baseline() -> None:
            gemm_out = torch.bmm(x_grouped, combined_t)
            result = torch.nn.functional.silu(gemm_out[..., : args.n].float())
            result = result * gemm_out[..., args.n :].float()
            baseline_out.copy_(result)

        def frost() -> None:
            flashinfer.frost_grouped_gemm1_swiglu(
                x, gate, up, offsets, scale, workspace, out=frost_out
            )

        for _ in range(args.warmup):
            baseline()
            frost()
        torch.cuda.synchronize()
        max_abs = (
            (baseline_out.reshape(tokens, args.n).float() - frost_out.float())
            .abs()
            .max()
            .item()
        )

        candidates: dict[str, Callable[[], None]] = {
            "baseline": baseline,
            "frost": frost,
        }
        graphs = []
        if args.cuda_graph:
            baseline_graph = _capture(baseline)
            frost_graph = _capture(frost)
            graphs.extend((baseline_graph, frost_graph))
            candidates = {
                "baseline": baseline_graph.replay,
                "frost": frost_graph.replay,
            }
        iterations = (
            args.iterations if tokens <= 1536 else max(10, args.iterations // 2)
        )
        medians = _interleaved_medians(
            candidates, rounds=args.rounds, iterations=iterations
        )
        print(
            f"S={tokens:5d} M/expert={group_m:4d} "
            f"baseline={medians['baseline']:8.4f} ms "
            f"frost={medians['frost']:8.4f} ms "
            f"speedup={medians['baseline'] / medians['frost']:6.2f}x "
            f"max_abs={max_abs:g}"
        )
        # Keep captured allocations alive through all replay measurements.
        del graphs


if __name__ == "__main__":
    main()
