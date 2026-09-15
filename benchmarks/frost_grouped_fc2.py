"""BF16 FC2-only PoC benchmark; NOT a full-MoE or best-backend comparison.

Reports every packaged FC2 config against a per-expert torch.mm loop, with
CUDA Graph replay removing Python-loop launch overhead from both sides.
Routing, FC1 and finalization are excluded. Scheduler reset is included.
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch

from flashinfer.experimental.frost_selected_kernels.fc2 import (
    PreparedFc2,
    matching_kernels,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", default="8192,12288")
    parser.add_argument("--experts", type=int, default=12)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--intermediate", type=int, default=3072)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--replays", type=int, default=30)
    args = parser.parse_args()
    torch.manual_seed(71)
    print(json.dumps(dict(gpu=torch.cuda.get_device_name(), **vars(args))), flush=True)
    e, h, i = args.experts, args.hidden, args.intermediate
    w = torch.randn(e, h, i, dtype=torch.bfloat16, device="cuda") * 0.02
    for s in map(int, args.tokens.split(",")):
        for distribution in ("uniform", "skew"):
            if distribution == "uniform":
                starts = [s * j // e for j in range(e)]
            else:
                # 50% on expert zero; remainder spread across all others.
                starts = [0] + [
                    s // 2 + (s - s // 2) * j // (e - 1) for j in range(e - 1)
                ]
            offsets = torch.tensor(starts, dtype=torch.int32, device="cuda")
            x = torch.randn(s, i, dtype=torch.bfloat16, device="cuda") * 0.1
            ref = torch.empty(s, h, dtype=torch.bfloat16, device="cuda")
            out = torch.empty_like(ref)
            blocks = [
                (x[a:b], w[j].T, ref[a:b])
                for j, (a, b) in enumerate(zip(starts, starts[1:] + [s], strict=True))
            ]

            def baseline():
                for a, b, c in blocks:
                    torch.mm(a, b, out=c)

            baseline()
            reference = ref.clone()
            kernels = matching_kernels(s, h, i, e, x.device)
            if not kernels:
                raise RuntimeError("no packaged FC2 artifact for this shape")
            plans = [PreparedFc2(k, x, w, offsets, out) for k in kernels]
            launches = [("torch_mm_per_expert", baseline)] + [
                (p.kernel.artifact_id, p.run) for p in plans
            ]
            graphs = {}
            errors = {}
            for label, launch in launches:
                for _ in range(3):
                    launch()
                current = ref if label == "torch_mm_per_expert" else out
                torch.testing.assert_close(current, reference, atol=2e-3, rtol=1e-2)
                errors[label] = (
                    (current.float() - reference.float()).norm()
                    / reference.float().norm()
                ).item()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    launch()
                graphs[label] = graph
            torch.cuda.synchronize()
            samples = {label: [] for label in graphs}
            labels = list(graphs)
            for r in range(args.rounds):
                ordered = labels[r % len(labels) :] + labels[: r % len(labels)]
                if r % 2:
                    ordered.reverse()
                for label in ordered:
                    begin, end = (
                        torch.cuda.Event(enable_timing=True) for _ in range(2)
                    )
                    begin.record()
                    for _ in range(args.replays):
                        graphs[label].replay()
                    end.record()
                    end.synchronize()
                    samples[label].append(begin.elapsed_time(end) / args.replays)
            timings = {
                label: statistics.median(vals) for label, vals in samples.items()
            }
            print(
                json.dumps(
                    dict(
                        tokens=s,
                        distribution=distribution,
                        milliseconds=timings,
                        relative_l2=errors,
                    )
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
