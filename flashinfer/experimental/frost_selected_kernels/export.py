"""Build-box exporter for FROST BF16 grouped FC1 + SwiGLU and FC2 kernels."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import re
from pathlib import Path
from typing import Any


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_graph(
    s: int,
    n: int,
    k: int,
    experts: int,
    groups: int,
    op: str = "grouped_gemm1_swiglu",
):
    import cudnn

    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    token = graph.tensor(
        name="token",
        dim=[1, s, k],
        stride=[s * k, k, 1],
        data_type=cudnn.data_type.BFLOAT16,
    )
    gate = graph.tensor(
        name="gate_weight",
        dim=[experts, k, n],
        stride=[k * n, 1, k],
        data_type=cudnn.data_type.BFLOAT16,
    )
    if op == "grouped_gemm2":
        offsets = graph.tensor(
            name="first_token_offset",
            dim=[groups, 1, 1],
            stride=[1, 1, 1],
            data_type=cudnn.data_type.INT32,
        )
        output = graph.moe_grouped_matmul(
            token,
            gate,
            offsets,
            mode=cudnn.moe_grouped_matmul_mode.NONE,
            compute_data_type=cudnn.data_type.FLOAT,
            name="fc2",
        )
        output.set_data_type(cudnn.data_type.BFLOAT16).set_output(True)
        return graph
    if op != "grouped_gemm1_swiglu":
        raise ValueError(f"unsupported export op {op!r}")
    up = graph.tensor(
        name="up_weight",
        dim=[experts, k, n],
        stride=[k * n, 1, k],
        data_type=cudnn.data_type.BFLOAT16,
    )
    offsets = graph.tensor(
        name="first_token_offset",
        dim=[groups, 1, 1],
        stride=[1, 1, 1],
        data_type=cudnn.data_type.INT32,
    )
    scale = graph.tensor(
        name="scale",
        dim=[1, 1, 1],
        stride=[1, 1, 1],
        data_type=cudnn.data_type.FLOAT,
    )
    gate_out = graph.moe_grouped_matmul(
        token,
        gate,
        offsets,
        mode=cudnn.moe_grouped_matmul_mode.NONE,
        compute_data_type=cudnn.data_type.FLOAT,
        name="gate_gemm",
    )
    up_out = graph.moe_grouped_matmul(
        token,
        up,
        offsets,
        mode=cudnn.moe_grouped_matmul_mode.NONE,
        compute_data_type=cudnn.data_type.FLOAT,
        name="up_gemm",
    )
    activated = graph.swish(input=gate_out, name="silu")
    swiglu = graph.mul(a=activated, b=up_out, name="swiglu")
    output = graph.mul(a=swiglu, b=scale, name="scale_output")
    output.set_data_type(cudnn.data_type.BFLOAT16).set_output(True)
    return graph


def _export_object(compiled: Any, path: Path, symbol: str) -> None:
    temporary = path.with_suffix(f".o.tmp.{os.getpid()}")
    try:
        export = compiled._launchable.export_to_c
        parameters = inspect.signature(export).parameters
        kwargs: dict[str, Any] = {}
        if "function_name" in parameters:
            kwargs["function_name"] = symbol
        if "export_only_tvm_ffi_symbols" in parameters:
            kwargs["export_only_tvm_ffi_symbols"] = True
        if "function_name" in parameters:
            export(str(temporary), **kwargs)
        else:
            export(str(temporary), symbol)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _compile_graph(graph: Any, config: Any, cta_group: int, scheduler: str) -> Any:
    """Call both the old and current FROST compiler APIs.

    Older FROST revisions selected the execution strategy with keyword
    arguments.  Current revisions encode ``cta_group`` in ``TileConfig`` and
    infer the scheduler from the selected template.
    """
    from cudnn.gemm.frost.compiler import jit_from_cudnn_graph

    parameters = inspect.signature(jit_from_cudnn_graph).parameters
    kwargs: dict[str, Any] = {"config": config}
    if "cta_group" in parameters:
        kwargs["cta_group"] = cta_group
    elif config.cta_group != cta_group:
        raise ValueError(
            f"tile config {config.name!r} has cta_group={config.cta_group}, "
            f"but --cta-group={cta_group} was requested"
        )
    if "scheduler" in parameters:
        kwargs["scheduler"] = scheduler
    elif scheduler != "clc":
        raise ValueError(
            "this FROST revision chooses the scheduler from the template; "
            "only --scheduler=clc is supported by this exporter"
        )
    # Newer Frost returns a reloaded callable even on a persistent-cache miss.
    # That callable cannot be re-exported under our sealed artifact symbol.
    # On the build box request the original exportable CuTe compiled object.
    cache_env = "CUDNN_FRONTEND_DISABLE_COMPILED_CACHE"
    previous = os.environ.get(cache_env)
    os.environ[cache_env] = "1"
    try:
        return jit_from_cudnn_graph(graph, **kwargs)
    finally:
        if previous is None:
            os.environ.pop(cache_env, None)
        else:
            os.environ[cache_env] = previous


def _select_template(chain: Any, config: Any, cta_group: int, scheduler: str) -> Any:
    from cudnn.gemm.frost.kernel_registry import select_template

    parameters = inspect.signature(select_template).parameters
    args = [chain, config]
    if "cta_group" in parameters:
        args.append(cta_group)
    if "scheduler" in parameters:
        args.append(scheduler)
    return select_template(*args)


def export_one(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from cudnn.gemm.frost.compiler import force_stg_epi
    from cudnn.gemm.frost.tile_config import CATALOG

    by_name = {config.name: config for config in CATALOG}
    try:
        config = by_name[args.config]
    except KeyError as exc:
        raise ValueError(
            f"unknown FROST tile config {args.config!r}; choose one of {sorted(by_name)}"
        ) from exc
    op = getattr(args, "op", "grouped_gemm1_swiglu")
    graph = _build_graph(args.s, args.n, args.k, args.experts, args.groups, op)
    with force_stg_epi(args.store_mode == "stg"):
        compiled = _compile_graph(graph, config, args.cta_group, args.scheduler)
    expected_gemms = 2 if op == "grouped_gemm1_swiglu" else 1
    if not compiled.chain.has_moe or compiled.chain.num_gemms != expected_gemms:
        raise RuntimeError(
            f"FROST did not compile {op} as {expected_gemms} grouped GEMM(s)"
        )
    store_modes = tuple(compiled.store_modes)
    if len(store_modes) != 1 or store_modes[0] not in ("stg", "tma"):
        raise RuntimeError(f"unexpected FROST output store modes: {store_modes!r}")
    actual_store_mode = store_modes[0]
    if args.store_mode == "tma" and actual_store_mode != "tma":
        raise RuntimeError(
            f"FROST did not produce the requested TMA-store kernel for {config.name}"
        )
    template = _select_template(compiled.chain, config, args.cta_group, args.scheduler)
    major, minor = torch.cuda.get_device_capability()
    arch = f"sm_{major}{minor}a"
    prefix = "grouped_swiglu" if op == "grouped_gemm1_swiglu" else "grouped_fc2"
    artifact_id = args.id or _slug(
        f"{prefix}_{arch}_e{args.experts}_n{args.n}_k{args.k}_"
        f"g{args.groups}_{config.name}_{args.cta_group}cta_{args.scheduler}_"
        f"{actual_store_mode}"
    )
    symbol = _slug(f"flashinfer_frost_{artifact_id}")

    output_dir = args.output_dir.resolve()
    objects = output_dir / "objects"
    objects.mkdir(parents=True, exist_ok=True)
    object_path = objects / f"{artifact_id}.o"
    if object_path.exists() and not args.replace:
        raise RuntimeError(
            f"artifact object {object_path} already exists; pass --replace to update it"
        )
    _export_object(compiled, object_path, symbol)
    tma_slots: frozenset[int] = getattr(compiled, "tma_slots", frozenset())
    launch_tail = ["scale", "output"] if 0 in tma_slots else ["output", "scale"]
    if op == "grouped_gemm2":
        launch_tail = ["output"]
    return {
        "id": artifact_id,
        "op": op,
        "arch": arch,
        "abi": f"frost_{op}_v1",
        "symbol": symbol,
        "object": {
            "path": object_path.relative_to(output_dir).as_posix(),
            "sha256": _sha256(object_path),
        },
        "workspace_bytes": int(compiled.workspace_bytes),
        "launch": {"tail": launch_tail},
        "contract": {
            "s": {"min": 1},
            "n": args.n,
            "k": args.k,
            "experts": args.experts,
            "groups": args.groups,
            "token_dtype": "bfloat16",
            "weight_dtype": "bfloat16",
            "output_dtype": "bfloat16",
            "activation": "silu(gate) * up" if expected_gemms == 2 else "identity",
        },
        "tactic": {
            "template": template.file,
            "tile": config.name,
            "cta_tile": {
                "m": int(config.cta_tile_m),
                "n": int(config.cta_tile_n),
                "k_bytes": int(config.cta_tile_k_bytes),
            },
            "cta_group": args.cta_group,
            "scheduler": args.scheduler,
            "store_mode": actual_store_mode,
        },
        "producer_revision": args.frost_revision,
    }


def _write_manifest(output_dir: Path, kernel: dict[str, Any], replace: bool) -> None:
    path = output_dir / "frost_selected_kernels.json"
    payload: dict[str, Any] = {
        "schema_version": 1,
        "producer": {"name": "frost"},
        "kernels": [],
    }
    if path.exists():
        payload = json.loads(path.read_text())
        if payload.get("schema_version") != 1:
            raise RuntimeError(f"refusing to update unsupported manifest {path}")
    kernels = list(payload.get("kernels", []))
    old = next(
        (i for i, item in enumerate(kernels) if item.get("id") == kernel["id"]),
        None,
    )
    if old is not None:
        if not replace:
            raise RuntimeError(
                f"artifact {kernel['id']!r} already exists; pass --replace to update it"
            )
        kernels[old] = kernel
    else:
        kernels.append(kernel)
    payload["kernels"] = sorted(kernels, key=lambda item: item["id"])
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--op",
        choices=("grouped_gemm1_swiglu", "grouped_gemm2"),
        default="grouped_gemm1_swiglu",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", required=True, help="FROST TileConfig name")
    parser.add_argument("--frost-revision", required=True)
    parser.add_argument("--cta-group", type=int, choices=(1, 2), default=2)
    parser.add_argument("--scheduler", choices=("clc", "static"), default="clc")
    parser.add_argument("--store-mode", choices=("tma", "stg"), default="tma")
    parser.add_argument("--id", help="stable artifact id (derived by default)")
    parser.add_argument("--s", type=int, default=1024)
    parser.add_argument(
        "--n", type=int, required=True, help="GEMM output width (FC2: hidden size)"
    )
    parser.add_argument(
        "--k",
        type=int,
        required=True,
        help="GEMM reduction width (FC2: intermediate size)",
    )
    parser.add_argument("--experts", type=int, required=True)
    parser.add_argument("--groups", type=int, help="defaults to --experts")
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    if args.groups is None:
        args.groups = args.experts
    kernel = export_one(args)
    _write_manifest(args.output_dir.resolve(), kernel, args.replace)
    print(f"exported {kernel['id']} to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
