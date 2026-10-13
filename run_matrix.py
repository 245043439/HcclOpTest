#!/usr/bin/env python3
"""Run an HCCL Cartesian benchmark matrix and append structured CSV results."""
import argparse
import csv
import itertools
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback


OPS = {"allreduce", "alltoall", "reducescatter", "allgather",
       "batch_send_recv", "broadcast", "send_recv"}
EXEC_MODES = {"eager", "ascend_ir", "aclgraph"}
EXPANSIONS = {"DEFAULT", "AI_CPU", "CCU_SCHED", "CCU_MS", "AIV"}
FIELDS = [
    "case_id", "operator", "size_bytes", "dtype", "group_size", "rank_start", "rank_end",
    "execution_mode", "requested_expansion_mode", "actual_expansion_mode", "input_bytes",
    "output_bytes", "warmup", "repeat", "mean_ms", "min_ms", "max_ms", "p50_ms", "p95_ms",
    "status", "reason", "log_path", "hardware", "cann_version", "torch_version",
    "torch_npu_version", "torchair_version",
]


def load_config(path):
    with open(path, encoding="utf-8") as stream:
        cfg = json.load(stream)
    required = ("operators", "sizes_bytes", "group_sizes", "execution_modes", "expansion_modes")
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError("missing config keys: " + ", ".join(missing))
    for key in required:
        if not isinstance(cfg[key], list) or not cfg[key]:
            raise ValueError(f"{key} must be a nonempty list")
    bad = {
        "operators": set(cfg["operators"]) - OPS,
        "execution_modes": set(cfg["execution_modes"]) - EXEC_MODES,
        "expansion_modes": set(cfg["expansion_modes"]) - EXPANSIONS,
    }
    for key, values in bad.items():
        if values:
            raise ValueError(f"unsupported {key}: {sorted(values)}")
    for key in ("sizes_bytes", "group_sizes"):
        if any(not isinstance(x, int) or isinstance(x, bool) for x in cfg[key]):
            raise ValueError(f"{key} entries must be integers")
    cfg.setdefault("warmup", 10)
    cfg.setdefault("repeat", 50)
    cfg.setdefault("case_timeout_sec", 900)
    cfg.setdefault("output_csv", "./results.csv")
    if cfg["warmup"] < 0 or cfg["repeat"] <= 0 or cfg["case_timeout_sec"] <= 0:
        raise ValueError("warmup must be >= 0; repeat and case_timeout_sec must be positive")
    return cfg


def make_cases(cfg):
    combos = itertools.product(cfg["operators"], cfg["sizes_bytes"], cfg["group_sizes"],
                               cfg["execution_modes"], cfg["expansion_modes"])
    cases = []
    for op, size, group, execution, expansion in combos:
        data = {"operator": op, "size_bytes": size, "group_size": group,
                "execution_mode": execution, "expansion_mode": expansion}
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
        data["case_id"] = __import__("hashlib").sha256(canonical.encode()).hexdigest()[:12]
        cases.append(data)
    return cases


def validate_case(case, world_size):
    size, group = case["size_bytes"], case["group_size"]
    if size <= 0:
        return "size_bytes must be positive"
    if size % 4:
        return "size_bytes must be divisible by float32 element size (4)"
    if group < 2 or group & (group - 1):
        return "group_size must be a power of two >= 2"
    if group > world_size or world_size % group:
        return f"group_size {group} must divide world size {world_size} and not exceed it"
    if case["operator"] in {"alltoall", "reducescatter", "allgather"}:
        elems = size // 4
        if elems % group:
            return f"element count {elems} must be divisible by group_size {group}"
    # Current Atlas 350 documentation identifies these expansion values as unsupported.
    if case["expansion_mode"] in {"AIV", "CCU_MS"}:
        return f"{case['expansion_mode']} is documented as unsupported on Atlas 350"
    return ""


def base_row(case, cfg, status, reason="", log_path="", start="", end=""):
    return {"case_id": case["case_id"], "operator": case["operator"],
            "size_bytes": case["size_bytes"], "dtype": "float32",
            "group_size": case["group_size"], "rank_start": start, "rank_end": end,
            "execution_mode": case["execution_mode"],
            "requested_expansion_mode": case["expansion_mode"],
            "actual_expansion_mode": "unknown", "input_bytes": "", "output_bytes": "",
            "warmup": cfg["warmup"], "repeat": cfg["repeat"],
            "mean_ms": "", "min_ms": "", "max_ms": "", "p50_ms": "", "p95_ms": "",
            "status": status, "reason": reason, "log_path": log_path,
            "hardware": "", "cann_version": "", "torch_version": "",
            "torch_npu_version": "", "torchair_version": ""}


def append_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())


def parse_nodes(value):
    nodes = [part.strip() for part in value.split(",") if part.strip()]
    if not nodes:
        raise ValueError("provide nodes through --nodes or HCCL_NODES")
    return nodes


def run_launcher(cmd, env, timeout_sec, log_path, case_id):
    """Stream launcher output to disk while keeping the operator informed."""
    start = time.monotonic()
    next_notice = start + 30
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                  start_new_session=True)
        while process.poll() is None:
            now = time.monotonic()
            if now - start >= timeout_sec:
                print(f"[{case_id}] timeout after {timeout_sec}s; stopping launcher process tree",
                      flush=True)
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                return 124
            if now >= next_notice:
                print(f"[{case_id}] still running ({int(now-start)}s); "
                      f"live cluster output: {log_path}", flush=True)
                next_notice = now + 30
            time.sleep(min(1, max(0.05, next_notice - now)))
        return process.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="JSON test matrix configuration")
    parser.add_argument("--dry-run", action="store_true", help="list planned combinations only")
    parser.add_argument("--nodes", default=os.environ.get("HCCL_NODES"),
                        help="comma-separated node IPs after master (or HCCL_NODES)")
    parser.add_argument("--launcher", default="./launch_cluster.sh")
    parser.add_argument("--log-dir", default="./hccl_matrix_logs")
    args = parser.parse_args()
    cfg = load_config(args.config)
    nproc = int(os.environ.get("NPROC_PER_NODE", "8"))
    if nproc <= 0:
        raise ValueError("NPROC_PER_NODE must be positive")
    nodes = parse_nodes(args.nodes) if args.nodes else []
    world_size = (len(nodes) + 1) * nproc if nodes else None
    cases = make_cases(cfg)
    if args.dry_run:
        print(f"combinations={len(cases)} world_size={world_size or 'unknown'}")
        for case in cases:
            reason = validate_case(case, world_size) if world_size else "world size unknown; runtime validation pending"
            print(f"{case['case_id']} {case['operator']} size={case['size_bytes']} group={case['group_size']} "
                  f"mode={case['execution_mode']} expansion={case['expansion_mode']}"
                  + (f" SKIP: {reason}" if reason and world_size else ""))
        return 0
    if not nodes:
        raise ValueError("execution requires --nodes or HCCL_NODES")
    output_csv = Path(cfg["output_csv"]).expanduser().resolve()
    log_root = Path(args.log_dir).expanduser().resolve()
    log_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    for index, case in enumerate(cases, 1):
        print(f"[{index}/{len(cases)}] {case['case_id']} {case['operator']} {case['size_bytes']}B "
              f"group={case['group_size']} {case['execution_mode']} {case['expansion_mode']}", flush=True)
        skip = validate_case(case, world_size)
        if skip:
            append_rows(output_csv, [base_row(case, cfg, "skipped", skip)])
            continue
        case_log_dir = log_root / case["case_id"]
        case_log_dir.mkdir(parents=True, exist_ok=True)
        host_result = case_log_dir / "result.json"
        try:
            host_result.unlink(missing_ok=True)
            env = os.environ.copy()
            env.update({
                "IMAGE_NAME": env.get("IMAGE_NAME", ""),
                "NPROC_PER_NODE": str(nproc), "SIZE_MB": str(case["size_bytes"] / 1024**2),
                "GROUP_SIZE": str(case["group_size"]), "WARMUP": str(cfg["warmup"]),
                "REPEAT": str(cfg["repeat"]), "TIMEOUT_SEC": str(cfg["case_timeout_sec"]),
                "LOG_DIR": str(case_log_dir), "MATRIX_OPERATOR": case["operator"],
                "MATRIX_EXECUTION_MODE": case["execution_mode"],
                "MATRIX_EXPANSION_MODE": case["expansion_mode"],
                "MATRIX_SIZE_BYTES": str(case["size_bytes"]),
                "CASE_RESULT_FILE": str(host_result),
                "BENCH_SCRIPT": env.get("BENCH_SCRIPT", str(Path(__file__).resolve().with_name("run_bench.py"))),
            })
            cmd = ["bash", args.launcher, *nodes]
            print(f"[{case['case_id']}] launcher log: {case_log_dir / 'launcher.log'}", flush=True)
            return_code = run_launcher(
                cmd, env, cfg["case_timeout_sec"] + 120,
                case_log_dir / "launcher.log", case["case_id"])
            if return_code != 0:
                status = "timeout" if return_code == 124 else "failed"
                rows = [base_row(case, cfg, status,
                                 f"launcher exited {return_code}", str(case_log_dir))]
            elif not host_result.exists():
                rows = [base_row(case, cfg, "failed", "launcher completed without structured result file",
                                 str(case_log_dir))]
            else:
                payload = json.loads(host_result.read_text(encoding="utf-8"))
                metadata = payload.get("metadata", {})
                rows = []
                for result in payload.get("results", []):
                    row = base_row(case, cfg, result.get("status", "failed"),
                                   result.get("reason", ""), str(case_log_dir),
                                   result.get("group_start", ""), result.get("group_end", ""))
                    row.update({"input_bytes": result.get("input_bytes", ""),
                                "output_bytes": result.get("output_bytes", ""),
                                "actual_expansion_mode": result.get("actual_expansion_mode", "unknown")})
                    row.update(metadata)
                    row.update(result.get("metrics", {}))
                    rows.append(row)
                if not rows:
                    rows = [base_row(case, cfg, "failed", "empty structured result", str(case_log_dir))]
            append_rows(output_csv, rows)
            if return_code in (69, 70):
                for remaining in cases[index:]:
                    append_rows(output_csv, [base_row(
                        remaining, cfg, "not_run",
                        "cluster preflight or cleanup failed; stopped matrix")])
                break
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            diagnostic = traceback.format_exc()
            (case_log_dir / "matrix.traceback.log").write_text(diagnostic, encoding="utf-8")
            print(diagnostic, file=sys.stderr, flush=True)
            append_rows(output_csv, [base_row(case, cfg, "failed", str(exc), str(case_log_dir))])
        except KeyboardInterrupt:
            append_rows(output_csv, [base_row(case, cfg, "failed", "interrupted by user", str(case_log_dir))])
            for remaining in cases[index:]:
                append_rows(output_csv, [base_row(remaining, cfg, "not_run", "matrix interrupted")])
            raise
    print(f"CSV: {output_csv} ({len(cases)} combinations, {time.time()-started:.1f}s)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        traceback.print_exc()
        sys.exit(2)
