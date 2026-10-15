# Repository Guidelines

## Project Structure & Module Organization

- `scripts/python/run_matrix.py` expands benchmark configurations, batches compatible cases, and manages CSV results and checkpoint recovery.
- `scripts/python/run_bench.py` executes distributed HCCL benchmarks using PyTorch and `torch_npu`.
- `scripts/python/query_npu_usage.py` queries worker nodes over SSH.
- `scripts/shell/` contains cluster launch, Docker setup, cleanup, and diagnostic collection scripts.
- `config/matrix_config.example.json` demonstrates matrix settings; `tests/` contains CPU-only regression tests. See `README.md` for deployment and result conventions.

## Build, Test, and Development Commands

There is no separate build step. Run commands from the repository root:

```bash
python3 -m unittest discover -s tests -v
bash -n scripts/shell/launch_cluster.sh scripts/shell/docker-gen.sh
python3 scripts/python/run_matrix.py --config config/matrix_config.example.json --dry-run --nodes 10.0.0.2,10.0.0.3
```

These run regression tests, check shell syntax, and preview pending matrix cases respectively. Dry runs honor existing results; add `--no-resume` to preview all cases.

For actual execution, remove `--dry-run` and set `IMAGE_NAME`, `DOCKER_SCRIPT`, and `BENCH_SCRIPT` as documented in the README. Hardware benchmarks run on the target servers; local NPU or CANN availability is not a prerequisite for development.

## Coding Style & Naming Conventions

Use four-space Python indentation, `snake_case` functions and variables, and `UPPER_CASE` constants. Follow surrounding Bash formatting, quote expansions, and preserve strict error handling. Keep configuration sizes in bytes and existing latency results in milliseconds; label any additional units explicitly. No formatter or linter configuration is currently checked in.

## Testing Guidelines

Use standard-library `unittest`, with files named `test_*.py` and methods named `test_*`. Keep unit tests independent of NPU, Docker, and SSH by using mocks and temporary directories. Add focused regressions for topology, batching, recovery, and diagnostic changes. No numerical coverage threshold is configured. Report target-server validation separately, including topology, dtype, execution mode, software versions, warmup, and repeat counts.

## Commit & Pull Request Guidelines

History uses short, action-oriented subjects such as `add comm_layer_list` and `fix send_recv rank_list`; no mandatory commit prefix is established. Keep commits focused. PR descriptions should explain the problem, resulting behavior, and checks performed. Link relevant issues and include commands or representative output for benchmark changes. Document any CSV schema, resume behavior, or configuration compatibility changes.

## Configuration & Operational Care

Keep credentials and private deployment details out of commits. Preserve checkpoints and diagnostic logs needed for recovery. Use a new output CSV or `--no-resume` when changing hardware, images, or software versions, since these changes are not part of resume matching.
