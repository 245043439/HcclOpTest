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
DTYPE_SIZES = {"float16": 2, "float32": 4, "bfloat16": 2,
               "int8": 1, "uint8": 1, "int16": 2, "int32": 4, "int64": 8}
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
    cfg.setdefault("dtype", "float32")
    if cfg["dtype"] not in DTYPE_SIZES:
        raise ValueError(f"unsupported dtype {cfg['dtype']!r}; choose one of {sorted(DTYPE_SIZES)}")
    if cfg["warmup"] < 0 or cfg["repeat"] <= 0 or cfg["case_timeout_sec"] <= 0:
        raise ValueError("warmup must be >= 0; repeat and case_timeout_sec must be positive")
    return cfg


def make_cases(cfg):
    combos = itertools.product(cfg["operators"], cfg["sizes_bytes"], cfg["group_sizes"],
                               cfg["execution_modes"], cfg["expansion_modes"])
    cases = []
    for op, size, group, execution, expansion in combos:
        data = {"operator": op, "size_bytes": size, "dtype": cfg["dtype"], "group_size": group,
                "execution_mode": execution, "expansion_mode": expansion}
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
        data["case_id"] = __import__("hashlib").sha256(canonical.encode()).hexdigest()[:12]
        cases.append(data)
    return cases


def group_cases(cases):
    groups = {}
    for case in cases:
        key = (case["group_size"], case["execution_mode"],
               case["expansion_mode"], case["dtype"])
        groups.setdefault(key, []).append(case)
    return list(groups.items())


def validate_case(case, world_size):
    size, group = case["size_bytes"], case["group_size"]
    if size <= 0:
        return "size_bytes must be positive"
    dtype = case["dtype"]
    elem_size = DTYPE_SIZES[dtype]
    if size % elem_size:
        return f"size_bytes must be divisible by {dtype} element size ({elem_size})"
    if group < 2 or group & (group - 1):
        return "group_size must be a power of two >= 2"
    if group > world_size or world_size % group:
        return f"group_size {group} must divide world size {world_size} and not exceed it"
    if case["operator"] in {"alltoall", "reducescatter", "allgather"}:
        elems = size // elem_size
        if elems % group:
            return f"element count {elems} must be divisible by group_size {group}"
    # Current Atlas 350 documentation identifies these expansion values as unsupported.
    if case["expansion_mode"] in {"AIV", "CCU_MS"}:
        return f"{case['expansion_mode']} is documented as unsupported on Atlas 350"
    return ""


def base_row(case, cfg, status, reason="", log_path="", start="", end=""):
    return {"case_id": case["case_id"], "operator": case["operator"],
            "size_bytes": case["size_bytes"], "dtype": case["dtype"],
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
    batches = group_cases(cases)
    if args.dry_run:
        valid = 0
        for key, batch in batches:
            reasons = {case["case_id"]: validate_case(case, world_size) if world_size else ""
                       for case in batch}
            count = sum(not reason for reason in reasons.values())
            valid += count
            print(f"batch group={key[0]} mode={key[1]} expansion={key[2]} dtype={key[3]} "
                  f"launches={1 if count else 0} cases={count}")
            for case in batch:
                reason = reasons[case["case_id"]]
                print(f"  {case['case_id']} {case['operator']} size={case['size_bytes']}B"
                      + (f" SKIP: {reason}" if reason else ""))
        print(f"combinations={len(cases)} batches={len(batches)} valid_cases={valid} "
              f"planned_launches={sum(any(not validate_case(c, world_size) for c in b) for _, b in batches) if world_size else len(batches)} "
              f"world_size={world_size or 'unknown'}")
        return 0
    if not nodes:
        raise ValueError("execution requires --nodes or HCCL_NODES")
    output_csv = Path(cfg["output_csv"]).expanduser().resolve()
    log_root = Path(args.log_dir).expanduser().resolve()
    log_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    stopped = False
    for batch_index, (key, batch) in enumerate(batches, 1):
        pending = []
        for case in batch:
            skip = validate_case(case, world_size)
            if skip:
                append_rows(output_csv, [base_row(case, cfg, "skipped", skip)])
            else:
                pending.append(case)
        if not pending:
            continue
        batch_id = f"batch_{batch_index:04d}_g{key[0]}_{key[1]}_{key[2]}_{key[3]}"
        attempt = 0
        while pending:
            attempt += 1
            run_dir = log_root / batch_id / f"attempt_{attempt}"
            run_dir.mkdir(parents=True, exist_ok=True)
            host_result = run_dir / "checkpoint.json"
            host_result.unlink(missing_ok=True)
            env = os.environ.copy()
            env.update({"IMAGE_NAME": env.get("IMAGE_NAME", ""),
                "NPROC_PER_NODE": str(nproc), "GROUP_SIZE": str(key[0]),
                "WARMUP": str(cfg["warmup"]), "REPEAT": str(cfg["repeat"]),
                "CASE_TIMEOUT_SEC": str(cfg["case_timeout_sec"]),
                "TIMEOUT_SEC": str(cfg["case_timeout_sec"] * (len(pending) + 1)),
                "LOG_DIR": str(run_dir),
                "MATRIX_EXECUTION_MODE": key[1], "MATRIX_EXPANSION_MODE": key[2],
                "MATRIX_DTYPE": key[3], "MATRIX_CASES_JSON": json.dumps(
                    [{"case_id": c["case_id"], "operator": c["operator"],
                      "size_bytes": c["size_bytes"]} for c in pending], separators=(",", ":")),
                "CASE_RESULT_FILE": str(host_result),
                "BENCH_SCRIPT": env.get("BENCH_SCRIPT", str(Path(__file__).resolve().with_name("run_bench.py")))})
            label = f"{batch_id}/attempt_{attempt}"
            print(f"[{label}] launching {len(pending)} cases; first={pending[0]['case_id']}", flush=True)
            try:
                return_code = run_launcher(["bash", args.launcher, *nodes], env,
                    int(env["TIMEOUT_SEC"]) + 120,
                    run_dir / "launcher.log", label)
                payload = None
                if host_result.exists():
                    payload = json.loads(host_result.read_text(encoding="utf-8"))
                completed = set()
                if payload is not None:
                    metadata = payload.get("metadata", {})
                    for result in payload.get("results", []):
                        case = next((item for item in pending if item["case_id"] == result.get("case_id")), None)
                        if case is None:
                            continue
                        completed.add(case["case_id"])
                        row = base_row(case, cfg, result.get("status", "failed"), result.get("reason", ""),
                                       str(run_dir), result.get("group_start", ""), result.get("group_end", ""))
                        row.update({"input_bytes": result.get("input_bytes", ""),
                                    "output_bytes": result.get("output_bytes", ""),
                                    "actual_expansion_mode": result.get("actual_expansion_mode", "unknown")})
                        row.update(metadata)
                        row.update(result.get("metrics", {}))
                        append_rows(output_csv, [row])
                if return_code == 0 and payload and payload.get("complete"):
                    break
                if return_code in (69, 70):
                    for case in pending:
                        if case["case_id"] not in completed:
                            append_rows(output_csv, [base_row(case, cfg, "not_run",
                                "cluster preflight or cleanup failed; stopped matrix", str(run_dir))])
                    stopped = True
                    break
                current = payload.get("current_case_id") if payload else None
                failed = next((case for case in pending if case["case_id"] == current
                               and case["case_id"] not in completed), None)
                if failed:
                    status = "timeout" if return_code == 124 else "failed"
                    append_rows(output_csv, [base_row(failed, cfg, status,
                        f"batch launcher exited {return_code} during case", str(run_dir))])
                    pending = [case for case in pending if case["case_id"] not in completed
                               and case["case_id"] != failed["case_id"]]
                else:
                    reason = "batch failed before a trustworthy current-case checkpoint"
                    for case in pending:
                        if case["case_id"] not in completed:
                            append_rows(output_csv, [base_row(case, cfg, "failed", reason, str(run_dir))])
                    break
            except KeyboardInterrupt:
                raise
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                diagnostic = traceback.format_exc()
                (run_dir / "matrix.traceback.log").write_text(diagnostic, encoding="utf-8")
                print(diagnostic, file=sys.stderr, flush=True)
                for case in pending:
                    append_rows(output_csv, [base_row(case, cfg, "failed", str(exc), str(run_dir))])
                break
        if stopped:
            for _, remaining_batch in batches[batch_index:]:
                for case in remaining_batch:
                    append_rows(output_csv, [base_row(case, cfg, "not_run", "cluster preflight or cleanup failed; stopped matrix")])
            break
    print(f"CSV: {output_csv} ({len(cases)} combinations, {time.time()-started:.1f}s)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        traceback.print_exc()
        sys.exit(2)
