#!/usr/bin/env python3
"""Minimal eager AllReduce benchmark and CPU-only msprof comparison."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import time


DTYPE_BYTES = {"float16": 2, "float32": 4, "bfloat16": 2}
CONFIG_KEYS = ("world_size", "size_bytes", "dtype", "warmup", "repeat", "execution_mode",
               "expansion_mode")


def positive_number(value, label, allow_zero=False):
    number = float(value)
    if not math.isfinite(number) or number < 0 or (number == 0 and not allow_zero):
        raise ValueError(f"{label}: expected a finite {'nonnegative' if allow_zero else 'positive'} value")
    return number


def run(args):
    if args.size_bytes <= 0 or args.size_bytes % DTYPE_BYTES[args.dtype]:
        raise ValueError("size-bytes must be positive and divisible by dtype element size")
    if args.warmup < 0 or args.repeat <= 0:
        raise ValueError("warmup must be nonnegative and repeat must be positive")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size < 2:
        raise ValueError("launch with torchrun and at least two ranks")
    args.output.mkdir(parents=True, exist_ok=True)
    result_path = args.output / f"rank_{rank}.json"
    if result_path.exists():
        raise ValueError(f"refusing to overwrite {result_path}; use a fresh output directory")
    if args.msprof:
        profile_dir = args.output.resolve() / "profiler" / f"rank_{rank}"
        profile_dir.mkdir(parents=True, exist_ok=False)
        command = ["msprof", f"--output={profile_dir}", "--task-time=on", "--hccl=on",
                   "--ai-core=off", "--type=text", sys.executable, str(Path(__file__).resolve()),
                   "run", "--profiled", "--output", str(args.output.resolve()),
                   "--size-bytes", str(args.size_bytes), "--dtype", args.dtype,
                   "--warmup", str(args.warmup), "--repeat", str(args.repeat)]
        return subprocess.run(command, check=False).returncode

    # Analysis and CLI help deliberately do not import any NPU dependencies.
    import torch
    import torch_npu
    import torch.distributed as dist

    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl")
    try:
        x = torch.zeros(args.size_bytes // DTYPE_BYTES[args.dtype],
                        dtype=getattr(torch, args.dtype), device=f"npu:{local_rank}")
        for _ in range(args.warmup):
            dist.all_reduce(x)
        torch.npu.synchronize()
        dist.barrier()
        samples = []
        for _ in range(args.repeat):
            dist.barrier()
            torch.npu.synchronize()
            start = time.perf_counter()
            dist.all_reduce(x)
            torch.npu.synchronize()
            samples.append((time.perf_counter() - start) * 1_000_000)
        payload = {
            "schema_version": 1, "rank": rank, "local_rank": local_rank,
            "world_size": world_size, "size_bytes": args.size_bytes, "dtype": args.dtype,
            "warmup": args.warmup, "repeat": args.repeat, "execution_mode": "eager",
            "expansion_mode": os.environ.get("HCCL_OP_EXPANSION_MODE", "DEFAULT"),
            "profiled": args.profiled, "samples_us": samples,
            "metadata": {"hostname": socket.gethostname(),
                         "hardware": getattr(torch.npu.get_device_properties(local_rank), "name", "unknown"),
                         "torch_version": torch.__version__,
                         "torch_npu_version": getattr(torch_npu, "__version__", "unknown"),
                         "cann_version": os.environ.get("CANN_VERSION", "unknown")},
        }
        with result_path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
        print(f"rank {rank}: saved {len(samples)} samples to {result_path}", flush=True)
    finally:
        dist.destroy_process_group()
    return 0


def load_run(directory, profiled):
    paths = sorted(directory.glob("rank_*.json"))
    if not paths:
        raise ValueError(f"{directory}: no rank results")
    ranks = {}
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        rank = data["rank"]
        if (type(rank) is not int or rank in ranks or path.name != f"rank_{rank}.json"
                or data["schema_version"] != 1 or data["profiled"] is not profiled):
            raise ValueError(f"{path}: invalid rank, schema, or profiling mode")
        for key in ("world_size", "size_bytes", "warmup", "repeat"):
            if type(data[key]) is not int:
                raise ValueError(f"{path}: {key} must be an integer")
        if (data["world_size"] < 2 or data["repeat"] <= 0 or data["warmup"] < 0
                or data["dtype"] not in DTYPE_BYTES or data["size_bytes"] <= 0
                or data["size_bytes"] % DTYPE_BYTES[data["dtype"]]
                or data["execution_mode"] != "eager"):
            raise ValueError(f"{path}: invalid benchmark configuration")
        if len(data["samples_us"]) != data["repeat"]:
            raise ValueError(f"{path}: sample count does not match repeat")
        data["samples_us"] = [positive_number(v, str(path)) for v in data["samples_us"]]
        ranks[rank] = data
    reference = next(iter(ranks.values()))
    if set(ranks) != set(range(reference["world_size"])):
        raise ValueError(f"{directory}: missing or unexpected ranks")
    for data in ranks.values():
        if any(data[key] != reference[key] for key in CONFIG_KEYS):
            raise ValueError(f"{directory}: configuration differs across ranks")
    return ranks


def time_column(headers, prefix):
    for unit, scale in (("us", 1), ("ms", 1000), ("ns", 0.001)):
        name = f"{prefix}({unit})"
        if name in headers:
            return name, scale
    raise ValueError(f"unsupported op_summary schema: missing {prefix} with us/ms/ns unit")


def read_communication(directory, warmup, repeat):
    """Read full communication rows, never sum internal kernel/task durations.

    msprof marks aggregated communication operations with N/A stream/task IDs.
    Real stream/task IDs identify internal tasks even when their names contain
    AllReduce. Reject unsupported exports rather than guessing their semantics.
    """
    paths = sorted(directory.rglob("op_summary*.csv"))
    if not paths:
        raise ValueError(f"{directory}: no op_summary CSV; export msprof text results first")
    events = []
    for path in paths:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            headers = reader.fieldnames or []
            required = {"Op Name", "Op Type", "Task Type", "Stream ID", "Task ID"}
            if not required.issubset(headers):
                raise ValueError(f"{path}: unsupported op_summary columns; need {sorted(required)}")
            start_key, start_scale = time_column(headers, "Task Start Time")
            duration_key, duration_scale = time_column(headers, "Task Duration")
            for row in reader:
                op_type = row["Op Type"].strip().lower().replace("_", "")
                if (row["Task Type"].strip().upper() != "COMMUNICATION"
                        or op_type not in ("allreduce", "hcomallreduce")
                        or row["Stream ID"].strip() != "N/A"
                        or row["Task ID"].strip() != "N/A"):
                    continue
                start = positive_number(row[start_key], f"{path}: start", allow_zero=True) * start_scale
                duration = positive_number(row[duration_key], f"{path}: duration") * duration_scale
                events.append({"start_us": start, "duration_us": duration,
                               "op_name": row["Op Name"], "source": str(path)})
    if len(events) != warmup + repeat:
        raise ValueError(f"{directory}: expected {warmup + repeat} full AllReduce events "
                         f"({warmup} warmup + {repeat} measured), found {len(events)}; "
                         "check full communication export, missing data, or duplicate exports")
    events.sort(key=lambda event: event["start_us"])
    if len({event["start_us"] for event in events}) != len(events):
        raise ValueError(f"{directory}: duplicate/ambiguous AllReduce timestamps")
    return events[warmup:]


def stats(values):
    return {"mean": statistics.fmean(values), "min": min(values), "max": max(values),
            "p50": statistics.median(values),
            "p95": sorted(values)[max(0, int(0.95 * len(values)) - 1)]}


def comparison_values(baseline, profiled, communication):
    return {"baseline_e2e_us": baseline, "profiled_e2e_us": profiled,
            "communication_us": communication,
            "e2e_minus_communication_us": profiled - communication,
            "e2e_minus_communication_pct": (profiled / communication - 1) * 100,
            "profiler_change_us": profiled - baseline,
            "profiler_change_pct": (profiled / baseline - 1) * 100}


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(args):
    baseline = load_run(args.baseline, False)
    profiled = load_run(args.profiled, True)
    if set(baseline) != set(profiled):
        raise ValueError("baseline and profiled runs have different ranks")
    for rank in baseline:
        if any(baseline[rank][key] != profiled[rank][key] for key in CONFIG_KEYS):
            raise ValueError(f"rank {rank}: baseline and profiled configurations differ")
        if (baseline[rank]["metadata"] != profiled[rank]["metadata"]
                or baseline[rank]["local_rank"] != profiled[rank]["local_rank"]):
            raise ValueError(f"rank {rank}: hardware, placement, or software metadata differ")
    config = {key: baseline[0][key] for key in CONFIG_KEYS}
    communication = {
        rank: read_communication(args.profiled / "profiler" / f"rank_{rank}",
                                 config["warmup"], config["repeat"])
        for rank in baseline
    }
    paired_rows, max_rows = [], []
    for iteration in range(config["repeat"]):
        base = {r: baseline[r]["samples_us"][iteration] for r in baseline}
        prof = {r: profiled[r]["samples_us"][iteration] for r in baseline}
        comm = {r: communication[r][iteration]["duration_us"] for r in baseline}
        for rank in sorted(baseline):
            event = communication[rank][iteration]
            paired_rows.append({"iteration": iteration, "rank": rank,
                                **comparison_values(base[rank], prof[rank], comm[rank]),
                                "communication_start_us": event["start_us"],
                                "op_name": event["op_name"], "source": event["source"]})
        max_rows.append({"iteration": iteration,
                         **comparison_values(max(base.values()), max(prof.values()), max(comm.values())),
                         "baseline_max_rank": max(base, key=base.get),
                         "profiled_max_rank": max(prof, key=prof.get),
                         "communication_max_rank": max(comm, key=comm.get)})
    metrics = {key: stats([row[key] for row in max_rows])
               for key in comparison_values(1, 1, 1)}
    summary = {
        "config": config, "metadata_by_rank": {r: baseline[r]["metadata"] for r in baseline},
        "metrics": metrics,
        "ratios_of_means_pct": {
            "e2e_minus_communication": (metrics["profiled_e2e_us"]["mean"] /
                                        metrics["communication_us"]["mean"] - 1) * 100,
            "profiler_change": (metrics["profiled_e2e_us"]["mean"] /
                                metrics["baseline_e2e_us"]["mean"] - 1) * 100},
        "notes": ["Latency statistics use the maximum rank duration in each iteration.",
                  "Maxima may come from different ranks; see rank_pairs.csv for same-rank gaps.",
                  "Baseline/profiled iterations are independent runs, matched only by ordinal.",
                  "Differences include waiting and profiling effects, not just Python overhead.",
                  "Percent denominators: communication for E2E gap; baseline E2E for profiler change."]}
    # Do not leave partial reports on validation failure, or mix reports from different runs.
    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output / "rank_pairs.csv", paired_rows)
    write_csv(args.output / "iterations.csv", max_rows)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"{'metric':36s} {'mean':>12s} {'min':>12s} {'max':>12s} {'p50':>12s} {'p95':>12s}")
    for name, values in metrics.items():
        print(f"{name:36s} " + " ".join(f"{v:12.3f}" for v in values.values()))
    print("\n" + "\n".join(summary["notes"]))
    print(f"\nReports: {args.output}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run", help="launch under torchrun, optionally wrap each rank with msprof")
    runner.add_argument("--size-bytes", type=int, default=1024 ** 2)
    runner.add_argument("--dtype", choices=DTYPE_BYTES, default="float32")
    runner.add_argument("--warmup", type=int, default=10)
    runner.add_argument("--repeat", type=int, default=50)
    runner.add_argument("--output", type=Path, required=True)
    runner.add_argument("--msprof", action="store_true")
    runner.add_argument("--profiled", action="store_true", help=argparse.SUPPRESS)
    analyzer = commands.add_parser("analyze", help="compare raw rank JSON and msprof op_summary CSV on CPU")
    analyzer.add_argument("--baseline", type=Path, required=True)
    analyzer.add_argument("--profiled", type=Path, required=True)
    analyzer.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        return run(args) if args.command == "run" else analyze(args)
    except (ValueError, KeyError, TypeError, OSError, csv.Error) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())
