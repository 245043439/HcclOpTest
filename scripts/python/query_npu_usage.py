#!/usr/bin/env python3
"""Query NPU utilization and process tables on remote Ascend hosts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
import re
import shlex
import subprocess
import sys

SSH_LOGIN_BANNER = "Authorized users only. All activities may be monitored and reported."


def remove_login_banner(output):
    return "\n".join(line for line in output.splitlines()
                      if line.strip() != SSH_LOGIN_BANNER)


def query_node(ip, ssh_user, connect_timeout, command_timeout):
    destination = f"{ssh_user}@{ip}"
    remote_command = (
        "if command -v npu-smi >/dev/null 2>&1; then npu-smi info; "
        "elif [ -x /usr/local/bin/npu-smi ]; then /usr/local/bin/npu-smi info; "
        "elif [ -x /usr/local/sbin/npu-smi ]; then /usr/local/sbin/npu-smi info; "
        "else echo 'npu-smi not found' >&2; exit 127; fi"
    )
    command = ["ssh", "-T", "-o", "BatchMode=yes",
               "-o", f"ConnectTimeout={connect_timeout}",
               "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=3",
               destination, f"bash -lc {shlex.quote(remote_command)}"]
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                timeout=command_timeout, check=False)
        return ip, result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        return ip, 124, stdout, f"SSH query timed out after {command_timeout}s\n{stderr}"
    except OSError as exc:
        return ip, 127, "", f"cannot start ssh: {exc}"


def summarize_processes(output):
    """Parse the process table printed by npu-smi info."""
    lines = output.splitlines()
    header = None
    processes = []
    no_process_npus = set()

    for line in lines:
        empty_match = re.search(r"No running processes found in NPU\s+(\d+)", line, re.I)
        if empty_match:
            no_process_npus.add(empty_match.group(1))
            continue

        if "|" not in line:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        normalized = [re.sub(r"\s+", " ", cell).lower() for cell in cells]
        if "npu id" in normalized and "process id" in normalized:
            header = normalized
            continue
        if header is None or len(cells) < 2:
            continue
        if not cells[0].isdigit() or not cells[1].isdigit():
            continue

        record = dict(zip(header, cells))
        processes.append(record)

    if processes:
        return "有进程", processes
    if header is not None and no_process_npus:
        return "无进程", []
    return "无法判断（未识别到进程表或明确的空进程提示）", []


def print_process_summary(status, stdout, show_processes=False):
    if status != 0:
        print(f"NPU 进程状态：无法判断（查询失败，退出码 {status}）")
        return

    summary, processes = summarize_processes(stdout)
    print(f"NPU 进程状态：{summary}")
    for process in processes if show_processes else ():
        fields = (
            ("npu id", "NPU"),
            ("process id", "PID"),
            ("process name", "进程"),
            ("process memory(mb)", "显存(MB)"),
            ("process id in container", "容器PID"),
        )
        details = [f"{label}={process[key]}" for key, label in fields
                   if key in process and process[key]]
        print("  " + " | ".join(details))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", required=True,
                        help="comma-separated node IPs, for example 10.0.0.1,10.0.0.2")
    parser.add_argument("--ssh-user", default=os.environ.get("SSH_USER", "root"),
                        help="SSH account (default: SSH_USER or root)")
    parser.add_argument("--connect-timeout", type=int, default=10,
                        help="SSH connection timeout in seconds")
    parser.add_argument("--timeout", type=int, default=30,
                        help="full query timeout per node in seconds")
    parser.add_argument("--verbose", action="store_true",
                        help="also print the full npu-smi info output")
    args = parser.parse_args()
    nodes = list(dict.fromkeys(part.strip() for part in args.nodes.split(",") if part.strip()))
    if not nodes:
        parser.error("--nodes must contain at least one IP")
    if args.connect_timeout <= 0 or args.timeout <= 0:
        parser.error("timeouts must be positive")

    with ThreadPoolExecutor(max_workers=len(nodes)) as pool:
        results = list(pool.map(lambda ip: query_node(
            ip, args.ssh_user, args.connect_timeout, args.timeout), nodes))

    groups = {"有进程": [], "无进程": [], "无法判断": []}
    processed = []
    for ip, status, stdout, stderr in results:
        stdout = remove_login_banner(stdout)
        stderr = remove_login_banner(stderr)
        if status == 0:
            summary, processes = summarize_processes(stdout)
        else:
            summary, processes = "无法判断", []
        group = summary if summary in ("有进程", "无进程") else "无法判断"
        groups[group].append(ip)
        processed.append((ip, status, stdout, stderr, summary, processes))

    for group in ("有进程", "无进程", "无法判断"):
        if not groups[group]:
            continue
        print(f"{group}（{len(groups[group])} 个节点）")
        for ip in groups[group]:
            print(f"  {ip}")

    for ip, status, stdout, stderr, summary, processes in processed:
        if status == 0 and summary != "无法判断（未识别到进程表或明确的空进程提示）":
            continue
        if not args.verbose and (status != 0 or summary.startswith("无法判断")):
            reason = stderr.strip() or summary
            if reason:
                print(f"  {ip}：{reason}", file=sys.stderr)

    if args.verbose:
        for ip, status, stdout, stderr, summary, _ in processed:
            print(f"===== {ip} | {'OK' if status == 0 else f'FAILED ({status})'} =====")
            if stdout:
                print(stdout, end="" if stdout.endswith("\n") else "\n")
            if stderr:
                print(stderr, file=sys.stderr, end="" if stderr.endswith("\n") else "\n")
            print_process_summary(status, stdout, show_processes=True)

    return 1 if any(status != 0 for _, status, *_ in processed) else 0


if __name__ == "__main__":
    sys.exit(main())
