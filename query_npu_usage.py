#!/usr/bin/env python3
"""Query NPU utilization and process tables on remote Ascend hosts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
import shlex
import subprocess
import sys


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
    args = parser.parse_args()
    nodes = list(dict.fromkeys(part.strip() for part in args.nodes.split(",") if part.strip()))
    if not nodes:
        parser.error("--nodes must contain at least one IP")
    if args.connect_timeout <= 0 or args.timeout <= 0:
        parser.error("timeouts must be positive")

    failed = False
    with ThreadPoolExecutor(max_workers=len(nodes)) as pool:
        results = list(pool.map(lambda ip: query_node(
            ip, args.ssh_user, args.connect_timeout, args.timeout), nodes))
    for ip, status, stdout, stderr in results:
        print(f"===== {ip} | {'OK' if status == 0 else f'FAILED ({status})'} =====")
        if stdout:
            print(stdout, end="" if stdout.endswith("\n") else "\n")
        if stderr:
            print(stderr, file=sys.stderr, end="" if stderr.endswith("\n") else "\n")
        failed |= status != 0
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
