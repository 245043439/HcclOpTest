#!/usr/bin/env python3
"""Run an HCCL Cartesian benchmark matrix and append structured CSV results."""
import argparse
import csv
import itertools
import json
import os
import re
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
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
LEGACY_FIELDS = FIELDS[:]
ENDPOINT_FIELDS = LEGACY_FIELDS + ["source_rank", "target_rank", "rank_stride"]
FIELDS = [field for field in LEGACY_FIELDS if field not in {"rank_start", "rank_end"}]
FIELDS.insert(5, "rank_list")


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
    cfg.setdefault("rank_stride", 8)
    if (not isinstance(cfg["rank_stride"], int) or isinstance(cfg["rank_stride"], bool)
            or cfg["rank_stride"] <= 0):
        raise ValueError("rank_stride must be a positive integer")
    if "dtype" not in cfg:
        raise ValueError("missing config key: dtype")
    if cfg["dtype"] not in DTYPE_SIZES:
        raise ValueError(f"unsupported dtype {cfg['dtype']!r}; choose one of {sorted(DTYPE_SIZES)}")
    if cfg["warmup"] < 0 or cfg["repeat"] <= 0 or cfg["case_timeout_sec"] <= 0:
        raise ValueError("warmup must be >= 0; repeat and case_timeout_sec must be positive")
    return cfg


def make_cases(cfg, node_count=None, nproc_per_node=None):
    combos = itertools.product(cfg["operators"], cfg["sizes_bytes"], cfg["group_sizes"],
                               cfg["execution_modes"], cfg["expansion_modes"])
    cases = []
    for op, size, group, execution, expansion in combos:
        data = {"operator": op, "size_bytes": size, "dtype": cfg["dtype"], "group_size": group,
                "execution_mode": execution, "expansion_mode": expansion}
        data["node_count"] = node_count
        data["nproc_per_node"] = nproc_per_node
        total_ranks = (node_count * nproc_per_node
                       if node_count is not None and nproc_per_node is not None else None)
        data["total_ranks"] = total_ranks
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
        digest = __import__("hashlib").sha256(canonical.encode()).hexdigest()[:12]
        nodes_label = node_count if node_count is not None else "x"
        procs_label = nproc_per_node if nproc_per_node is not None else "x"
        ranks_label = total_ranks if total_ranks is not None else "x"
        data["case_id"] = f"g{ranks_label}_n{nodes_label}_p{procs_label}_group{group}_{digest}"
        cases.append(data)
    return cases


def make_topology_cases(cfg, available_nodes, max_procs_per_node):
    cases = []
    for group in cfg["group_sizes"]:
        topologies = [(n, group // n) for n in range(min(available_nodes, max(group, 0)), 0, -1)
                      if group % n == 0 and 1 <= group // n <= max_procs_per_node]
        group_cfg = dict(cfg, group_sizes=[group])
        if not topologies:
            skipped = make_cases(group_cfg)
            for case in skipped:
                case["topology_error"] = (f"no topology for group_size {group} with "
                    f"{available_nodes} available nodes and at most {max_procs_per_node} ranks per node")
            cases.extend(skipped)
        for node_count, procs in topologies:
            cases.extend(make_cases(group_cfg, node_count, procs))
    return cases


def group_cases(cases):
    groups = {}
    for case in cases:
        key = (case["group_size"], case["execution_mode"],
               case["expansion_mode"], case["dtype"],
               case["node_count"], case["nproc_per_node"])
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
    if case.get("topology_error"):
        return case["topology_error"]
    if group != world_size:
        return f"group_size {group} must equal launch world size {world_size}"
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
            "group_size": case["group_size"],
            "rank_list": json.dumps(global_ranks(case, cfg, start, end)),
            "execution_mode": case["execution_mode"],
            "requested_expansion_mode": case["expansion_mode"],
            "actual_expansion_mode": "unknown", "input_bytes": "", "output_bytes": "",
            "warmup": cfg["warmup"], "repeat": cfg["repeat"],
            "mean_ms": "", "min_ms": "", "max_ms": "", "p50_ms": "", "p95_ms": "",
            "status": status, "reason": reason, "log_path": log_path,
            "hardware": "", "cann_version": "", "torch_version": "",
            "torch_npu_version": "", "torchair_version": ""}


def global_ranks(case, cfg, start="", end=""):
    """Map operator participants to global physical slot numbers."""
    procs = case["nproc_per_node"]
    if not procs or not case["total_ranks"]:
        return []
    stride = cfg.get("rank_stride", 8)
    if procs > stride:
        raise ValueError("rank_stride must be at least the number of processes per node")
    first = 0 if start == "" else int(start)
    last = case["total_ranks"] - 1 if end == "" else int(end)
    ranks = [(r // procs) * stride + r % procs for r in range(first, last + 1)]
    if case.get("operator") == "send_recv" and ranks:
        return [ranks[0], ranks[-1]]
    return ranks


def legacy_rank_list(row):
    """Old case IDs encode placement; never guess an unknown placement."""
    match = re.search(r"_n([0-9]+)_p([0-9]+)_", row.get("case_id", ""))
    if not match or not row.get("rank_start") or not row.get("rank_end"):
        return []
    try:
        nodes, procs = map(int, match.groups())
        return global_ranks({"operator": row.get("operator"),
                             "nproc_per_node": procs, "total_ranks": nodes * procs},
                            {"rank_stride": int(row.get("rank_stride") or 8)},
                            row["rank_start"], row["rank_end"])
    except (ValueError, TypeError):
        return []


def upgrade_csv(path):
    """Preserve existing results while converting an old CSV to rank_list."""
    temp_path = None
    try:
        with path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames == FIELDS:
                return
            if reader.fieldnames not in (LEGACY_FIELDS, ENDPOINT_FIELDS):
                raise ValueError(f"incompatible result CSV header: {path}; use a new output_csv")
            with tempfile.NamedTemporaryFile(mode="w", newline="", encoding="utf-8",
                                             dir=path.parent, delete=False) as target:
                temp_path = Path(target.name)
                writer = csv.DictWriter(target, fieldnames=FIELDS, extrasaction="ignore")
                writer.writeheader()
                for row in reader:
                    row["rank_list"] = json.dumps(legacy_rank_list(row))
                    writer.writerow(row)
                target.flush()
                os.fsync(target.fileno())
        os.chmod(temp_path, path.stat().st_mode & 0o777)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def append_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        upgrade_csv(path)
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())


def successful_cases(path, cases, cfg):
    """Resume only complete world-group results with matching measurement settings."""
    if not path.exists() or path.stat().st_size == 0:
        return set()
    by_id = {case["case_id"]: case for case in cases}
    completed = set()
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames not in (FIELDS, LEGACY_FIELDS, ENDPOINT_FIELDS):
            raise ValueError(f"incompatible result CSV header: {path}; use a new output_csv")
        for row in reader:
            case_id = row.get("case_id")
            case = by_id.get(case_id)
            if case is None or (row.get("warmup"), row.get("repeat")) != (
                    str(cfg["warmup"]), str(cfg["repeat"])):
                continue
            if "rank_list" in row:
                try:
                    recorded_ranks = json.loads(row["rank_list"])
                    ranks_match = recorded_ranks == global_ranks(case, cfg)
                    if case["operator"] == "send_recv" and not ranks_match:
                        # Older result files recorded the full communicator for send_recv.
                        ranks_match = recorded_ranks == global_ranks(
                            dict(case, operator=""), cfg)
                except (TypeError, ValueError):
                    ranks_match = False
            else:
                ranks_match = (row.get("rank_start") == "0"
                               and row.get("rank_end") == str(case["total_ranks"] - 1)
                               if case["total_ranks"] is not None else False)
            # The latest matching attempt wins, including a later failed rerun.
            completed.discard(case_id)
            if (None not in row and all(value is not None for value in row.values())
                    and row["status"] == "success" and case["total_ranks"] is not None
                    and ranks_match):
                completed.add(case_id)
    return completed


def result_row(case, cfg, result, metadata, run_dir):
    row = base_row(case, cfg, result.get("status", "failed"), result.get("reason", ""),
                   str(run_dir), result.get("group_start", ""), result.get("group_end", ""))
    row.update({key: result.get(key, "") for key in ("input_bytes", "output_bytes")})
    row["actual_expansion_mode"] = result.get("actual_expansion_mode", "unknown")
    row.update(metadata)
    row.update(result.get("metrics", {}))
    return row


def recover_checkpoints(log_root, output_csv, cases, cfg, completed, dry_run=False):
    """Import interrupted attempts only; consumed checkpoints must never override newer runs."""
    completed = set(completed)
    by_id = {case["case_id"]: case for case in cases}
    for manifest_path in sorted(log_root.glob("batch_*/attempt_*/attempt.json")):
        run_dir = manifest_path.parent
        if (run_dir / "consumed").exists():
            continue
        checkpoint = run_dir / "checkpoint.json"
        if not checkpoint.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
            if (manifest["output_csv"] != str(output_csv)
                    or manifest["warmup"] != cfg["warmup"]
                    or manifest["repeat"] != cfg["repeat"]):
                continue
            payload = json.loads(checkpoint.read_text())
            for result in payload["results"]:
                case_id = result["case_id"]
                case = by_id.get(case_id)
                if (case is None or case_id not in manifest["case_ids"] or case_id in completed
                        or case["total_ranks"] is None
                        or result.get("status") != "success"
                        or result.get("group_start") != 0
                        or result.get("group_end") != case["total_ranks"] - 1):
                    continue
                if not dry_run:
                    append_rows(output_csv, [result_row(case, cfg, result,
                                payload.get("metadata", {}), run_dir)])
                completed.add(case_id)
                print(f"Recovered {case_id} from {checkpoint}", flush=True)
            if not dry_run and set(manifest["case_ids"]).issubset(by_id):
                (run_dir / "consumed").touch()
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"Cannot recover {checkpoint}: {exc}", file=sys.stderr)
    return completed


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
        try:
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
        except KeyboardInterrupt:
            print("Interrupted; stopping cluster and collecting checkpoint...", flush=True)
            # Signal the launcher itself so its EXIT trap can stop container tasks
            # and copy rank 0's checkpoint before we return to the caller.
            process.terminate()
            try:
                process.wait(timeout=120)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                print("Cleanup interrupted or timed out; checkpoint collection may be incomplete",
                      file=sys.stderr)
            raise



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="JSON test matrix configuration")
    parser.add_argument("--dry-run", action="store_true", help="list planned combinations only")
    parser.add_argument("--nodes", default=os.environ.get("HCCL_NODES") or None,
                        help="worker IPs to use (all are remote workers; omitted uses local master)")
    parser.add_argument("--launcher", default="./scripts/shell/launch_cluster.sh")
    parser.add_argument("--log-dir", default="./hccl_matrix_logs")
    parser.add_argument("--no-resume", action="store_true",
                        help="rerun all cases even when successful results exist in output_csv")
    args = parser.parse_args()
    cfg = load_config(args.config)
    nproc = int(os.environ.get("NPROC_PER_NODE", "8"))
    if nproc <= 0:
        raise ValueError("NPROC_PER_NODE must be positive")
    remote_nodes_only = args.nodes is not None
    nodes = parse_nodes(args.nodes) if remote_nodes_only else []
    available_nodes = len(nodes) if remote_nodes_only else 1
    cases = make_topology_cases(cfg, available_nodes, nproc)
    if any(c["nproc_per_node"] and c["nproc_per_node"] > cfg["rank_stride"] for c in cases):
        raise ValueError("rank_stride must be at least the number of processes per node")
    output_csv = Path(cfg["output_csv"]).expanduser().resolve()
    log_root = Path(args.log_dir).expanduser().resolve()
    completed = successful_cases(output_csv, cases, cfg)
    completed = recover_checkpoints(log_root, output_csv, cases, cfg, completed, args.dry_run)
    if args.no_resume:
        completed = set()
    print(f"Resume: {len(completed)} successful cases skipped; "
          f"{len(cases) - len(completed)} combinations remaining", flush=True)
    batches = group_cases([case for case in cases if case["case_id"] not in completed])
    if args.dry_run:
        valid = 0
        for key, batch in batches:
            reasons = {case["case_id"]: validate_case(case, case["total_ranks"])
                       for case in batch}
            count = sum(not reason for reason in reasons.values())
            valid += count
            print(f"batch group={key[0]} nodes={key[4]} procs_per_node={key[5]} mode={key[1]} expansion={key[2]} dtype={key[3]} "
                  f"launches={1 if count else 0} cases={count}")
            for case in batch:
                reason = reasons[case["case_id"]]
                print(f"  {case['case_id']} {case['operator']} size={case['size_bytes']}B"
                      + f" rank_list={global_ranks(case, cfg)}"
                      + (f" SKIP: {reason}" if reason else ""))
        print(f"combinations={len(cases)} batches={len(batches)} valid_cases={valid} "
              f"planned_launches={sum(any(not validate_case(c, c['total_ranks']) for c in b) for _, b in batches)} "
              f"available_nodes={available_nodes} max_procs_per_node={nproc}")
        return 0
    missing_scripts = [name for name in ("DOCKER_SCRIPT", "BENCH_SCRIPT")
                       if not os.environ.get(name)]
    if missing_scripts:
        raise ValueError("set required script path environment variables: "
                         + ", ".join(missing_scripts))
    log_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    stopped = False
    for batch_index, (key, batch) in enumerate(batches, 1):
        pending = []
        for case in batch:
            skip = validate_case(case, case["total_ranks"])
            if skip:
                append_rows(output_csv, [base_row(case, cfg, "skipped", skip)])
            else:
                pending.append(case)
        if not pending:
            continue
        node_count, batch_procs = key[4], key[5]
        selected_nodes = nodes[:node_count] if remote_nodes_only else []
        batch_id = (f"batch_{batch_index:04d}_g{key[0]}_n{node_count}_p{batch_procs}_group{key[0]}_"
                    f"{key[1]}_{key[2]}_{key[3]}")
        attempt = 0
        while pending:
            attempt += 1
            run_dir = log_root / batch_id / f"attempt_{attempt}"
            while run_dir.exists():
                attempt += 1
                run_dir = log_root / batch_id / f"attempt_{attempt}"
            run_dir.mkdir(parents=True)
            host_result = run_dir / "checkpoint.json"
            with (run_dir / "attempt.json").open("w", encoding="utf-8") as stream:
                json.dump({"output_csv": str(output_csv), "warmup": cfg["warmup"],
                           "repeat": cfg["repeat"], "case_ids": [c["case_id"] for c in pending]}, stream)
                stream.flush()
                os.fsync(stream.fileno())
            env = os.environ.copy()
            env.update({"IMAGE_NAME": env.get("IMAGE_NAME", ""),
                "NPROC_PER_NODE": str(batch_procs), "GROUP_SIZE": str(key[0]),
                "WARMUP": str(cfg["warmup"]), "REPEAT": str(cfg["repeat"]),
                "CASE_TIMEOUT_SEC": str(cfg["case_timeout_sec"]),
                "TIMEOUT_SEC": str(cfg["case_timeout_sec"] * (len(pending) + 1)),
                "LOG_DIR": str(run_dir),
                "MATRIX_EXECUTION_MODE": key[1], "MATRIX_EXPANSION_MODE": key[2],
                "MATRIX_DTYPE": key[3], "MATRIX_CASES_JSON": json.dumps(
                    [{"case_id": c["case_id"], "operator": c["operator"],
                      "size_bytes": c["size_bytes"]} for c in pending], separators=(",", ":")),
                "CASE_RESULT_FILE": str(host_result),
                "HCCL_NODES_ONLY": "1" if remote_nodes_only else "0",
                "BENCH_SCRIPT": env["BENCH_SCRIPT"],
                "DOCKER_SCRIPT": env["DOCKER_SCRIPT"]})
            label = f"{batch_id}/attempt_{attempt}"
            print(f"[{label}] launching {len(pending)} cases; first={pending[0]['case_id']}", flush=True)
            try:
                return_code = run_launcher(["bash", args.launcher, *selected_nodes], env,
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
                        append_rows(output_csv, [result_row(case, cfg, result, metadata, run_dir)])
                    (run_dir / "consumed").touch()
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
                recover_checkpoints(log_root, output_csv, cases, cfg,
                                    successful_cases(output_csv, cases, cfg))
                print("Interrupted. Run the same command to resume.", flush=True)
                return 130
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
