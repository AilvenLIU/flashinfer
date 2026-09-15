"""Benchmark packaged FROST grouped GEMM1 + SwiGLU tactics on SM100.

The default model dimensions follow FROST's own grouped SwiGLU benchmark.  The
token sweep covers decode-sized local work through the MegaMoE crossover, and
the two routing distributions expose both balanced and hot-expert behavior.
"""

from __future__ import annotations

import argparse
import statistics

import torch

from flashinfer.experimental.frost_selected_kernels.runtime import (
    FrostGroupedGemm1SwiGLURunner,
    matching_kernels,
    workspace_size,
)


def _csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(","))


def _offsets(tokens: int, experts: int, distribution: str) -> torch.Tensor:
    if distribution == "uniform":
        values = [(i * tokens) // experts for i in range(experts)]
    else:
        base = torch.tensor([48, 16, 9, 7, 5, 4, 3, 2, 2, 2, 1, 1], dtype=torch.float64)
        if experts != base.numel():
            raise ValueError("the default skew distribution requires 12 experts")
        starts = torch.cat((torch.zeros(1, dtype=torch.float64), base.cumsum(0)[:-1]))
        values = torch.floor(starts * tokens / base.sum()).to(torch.int32).tolist()
    return torch.tensor(values, device="cuda", dtype=torch.int32)


def _label(kernel) -> str:
    tile = kernel.tactic_metadata["cta_tile"]
    return (
        f"M{tile['m']}N{tile['n']}K{tile['k_bytes']}B-"
        f"{kernel.tactic_metadata['cta_group']}cta-"
        f"{kernel.tactic_metadata['store_mode']}"
    )


def _time_ms(runner, inputs, tactic, *, iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        runner(inputs=inputs, tactic=tactic)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experts", type=int, default=12)
    parser.add_argument("--n", type=int, default=3072)
    parser.add_argument("--k", type=int, default=7168)
    parser.add_argument(
        "--tokens", type=_csv_ints, default=_csv_ints("24,96,384,1536,6144,12288")
    )
    parser.add_argument(
        "--distributions", choices=("uniform", "skew", "both"), default="both"
    )
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=7)
    args = parser.parse_args()

    torch.manual_seed(41)
    gate = (
        torch.randn(args.experts, args.n, args.k, device="cuda", dtype=torch.bfloat16)
        * 0.02
    ).contiguous()
    up = (torch.randn_like(gate) * 0.02).contiguous()
    scale = torch.tensor([0.75], device="cuda", dtype=torch.float32)
    runner = FrostGroupedGemm1SwiGLURunner()
    distributions = (
        ("uniform", "skew") if args.distributions == "both" else (args.distributions,)
    )

    summary: dict[str, list[float]] = {}
    for distribution in distributions:
        for tokens in args.tokens:
            x = (
                torch.randn(tokens, args.k, device="cuda", dtype=torch.bfloat16) * 0.02
            ).contiguous()
            offsets = _offsets(tokens, args.experts, distribution)
            out = torch.empty(tokens, args.n, device="cuda", dtype=torch.bfloat16)
            kernels = matching_kernels(x, gate, up, offsets, scale, out)
            if not kernels:
                raise RuntimeError("no packaged FROST artifact matches the benchmark")
            workspace = torch.empty(
                workspace_size(x, gate, up, offsets, scale, out),
                device="cuda",
                dtype=torch.uint8,
            )
            inputs = [x, gate, up, offsets, scale, out, workspace]
            reference = None
            for kernel in kernels:
                runner(inputs=inputs, tactic=kernel.tactic)
                torch.cuda.synchronize()
                current = out.clone()
                if reference is None:
                    reference = current
                elif not torch.equal(current, reference):
                    error = (current.float() - reference.float()).abs().max().item()
                    raise RuntimeError(f"{_label(kernel)} differs by {error}")
                for _ in range(args.warmup):
                    runner(inputs=inputs, tactic=kernel.tactic)
            torch.cuda.synchronize()

            # Rotate and alternate the tactic order so clock/thermal drift does
            # not systematically favor the first or last manifest entry.
            iterations = (
                args.iterations if tokens <= 1536 else max(10, args.iterations // 2)
            )
            samples = {_label(kernel): [] for kernel in kernels}
            for round_index in range(args.rounds):
                offset = round_index % len(kernels)
                ordered = kernels[offset:] + kernels[:offset]
                if round_index % 2:
                    ordered = list(reversed(ordered))
                for kernel in ordered:
                    samples[_label(kernel)].append(
                        _time_ms(
                            runner,
                            inputs,
                            kernel.tactic,
                            iterations=iterations,
                        )
                    )
            rows = [
                (statistics.median(elapsed), label)
                for label, elapsed in samples.items()
            ]
            rows.sort()
            best = rows[0][0]
            print(f"S={tokens:5d} distribution={distribution}")
            for elapsed, label in rows:
                ratio = elapsed / best
                summary.setdefault(label, []).append(ratio)
                print(f"  {label:25s} {elapsed:8.4f} ms  {ratio:6.3f}x")

    print("summary")
    for label, ratios in sorted(summary.items()):
        wins = sum(ratio <= 1.01 for ratio in ratios)
        print(
            f"  {label:25s} wins<=1%={wins}/{len(ratios)} "
            f"geomean={statistics.geometric_mean(ratios):.3f}x "
            f"worst={max(ratios):.3f}x"
        )


if __name__ == "__main__":
    main()
