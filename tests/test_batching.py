"""CPU-only checks for matrix batching and checkpoint recovery."""
import csv
import ast
import importlib.util
import io
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("run_matrix", ROOT / "scripts" / "python" / "run_matrix.py")
matrix = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(matrix)


class BatchingTests(unittest.TestCase):
    def test_preflight_skips_batch_but_cleanup_stops_matrix(self):
        for failure_code in (69, 70):
            with self.subTest(failure_code=failure_code), tempfile.TemporaryDirectory() as directory:
                path = self.config(directory)
                cfg = json.loads(path.read_text())
                cfg["group_sizes"] = [8, 16]
                path.write_text(json.dumps(cfg))
                nodes = [f"10.0.0.{index}" for index in range(1, 9)]
                launches = []

                def launch(cmd, env, timeout, log_path, label):
                    launches.append((int(env["GROUP_SIZE"]), len(cmd[2:])))
                    # Every eight-node placement includes the unavailable node.
                    if len(cmd[2:]) == 8:
                        return failure_code
                    batch = json.loads(env["MATRIX_CASES_JSON"])
                    Path(env["CASE_RESULT_FILE"]).write_text(json.dumps({
                        "complete": True, "results": [dict(
                            case, status="success", group_start=0,
                            group_end=int(env["GROUP_SIZE"]) - 1) for case in batch]}))
                    return 0

                argv = ["run_matrix.py", "--config", str(path), "--nodes", ",".join(nodes),
                        "--log-dir", str(Path(directory) / "logs")]
                with mock.patch("sys.argv", argv), mock.patch.dict(os.environ, {
                        "NPROC_PER_NODE": "8", "DOCKER_SCRIPT": "docker-gen.sh",
                        "BENCH_SCRIPT": "run_bench.py"}), \
                     mock.patch.object(matrix, "run_launcher", side_effect=launch):
                    self.assertEqual(matrix.main(), 0)
                with Path(cfg["output_csv"]).open(newline="") as stream:
                    rows = list(csv.DictReader(stream))
                cases = matrix.make_topology_cases(matrix.load_config(path), 8, 8)
                self.assertEqual(len(rows), len(cases))
                by_id = {row["case_id"]: row for row in rows}
                self.assertEqual(len(by_id), len(cases))
                if failure_code == 69:
                    self.assertEqual(launches, [(8, 8), (8, 4), (8, 2), (8, 1),
                                                (16, 8), (16, 4), (16, 2)])
                    for case in cases:
                        self.assertEqual(by_id[case["case_id"]]["status"],
                                         "not_run" if case["node_count"] == 8 else "success")
                else:
                    self.assertEqual(launches, [(8, 8)])
                    self.assertTrue(all(row["status"] == "not_run" for row in rows))

    def test_restart_retries_unsuccessful_cases_and_preserves_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.config(directory)
            cfg = matrix.load_config(path)
            cases = matrix.make_topology_cases(cfg, 1, 2)
            output = Path(cfg["output_csv"])
            matrix.append_rows(output, [matrix.base_row(c, cfg, status, start=0, end=1)
                for c, status in zip(cases, ["success", "failed", "timeout", "not_run"])])
            launches, checkpoints = [], []

            def launch(cmd, env, timeout, log_path, label):
                batch = json.loads(env["MATRIX_CASES_JSON"])
                launches.append([c["case_id"] for c in batch])
                checkpoint = Path(env["CASE_RESULT_FILE"])
                checkpoints.append(checkpoint)
                checkpoint.write_text(json.dumps({"complete": True, "results": [
                    dict(c, status="success", group_start=0, group_end=1) for c in batch]}))
                return 0

            argv = ["run_matrix.py", "--config", str(path), "--log-dir", str(Path(directory) / "logs")]
            with mock.patch.dict(os.environ, {"NPROC_PER_NODE": "2", "HCCL_NODES": ""}), \
                 mock.patch.object(matrix, "run_launcher", side_effect=launch):
                with mock.patch("sys.argv", argv):
                    self.assertEqual(matrix.main(), 0)
                    contents = output.read_text()
                    self.assertEqual(matrix.main(), 0)
                    self.assertEqual(output.read_text(), contents)
                self.assertEqual(launches, [[c["case_id"] for c in cases[1:]]])
                old_checkpoint = checkpoints[0].read_text()
                with mock.patch("sys.argv", argv + ["--no-resume"]):
                    self.assertEqual(matrix.main(), 0)
                self.assertEqual(launches[-1], [c["case_id"] for c in cases])
                self.assertNotEqual(checkpoints[0], checkpoints[1])
                self.assertEqual(checkpoints[0].read_text(), old_checkpoint)
                with mock.patch("sys.argv", argv + ["--dry-run"]), \
                     mock.patch("sys.stdout", new_callable=io.StringIO) as stdout:
                    self.assertEqual(matrix.main(), 0)
                self.assertIn("planned_launches=0", stdout.getvalue())
                self.assertEqual(len(launches), 2)

    def test_resume_requires_matching_settings_and_complete_latest_result(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = matrix.load_config(self.config(directory))
            cases = matrix.make_topology_cases(cfg, 1, 2)
            path = Path(cfg["output_csv"])
            self.assertEqual(matrix.successful_cases(path, cases, cfg), set())
            matrix.append_rows(path, [matrix.base_row(c, cfg, "success", start=0, end=1)
                                      for c in cases])
            self.assertEqual(matrix.successful_cases(path, cases, cfg), {c["case_id"] for c in cases})
            for key in ("warmup", "repeat"):
                self.assertEqual(matrix.successful_cases(path, cases, dict(cfg, **{key: cfg[key] + 1})), set())
            other_topology = matrix.make_topology_cases(cfg, 2, 1)
            self.assertEqual(matrix.successful_cases(path, other_topology, cfg), set())
            matrix.append_rows(path, [matrix.base_row(cases[0], cfg, "failed"),
                                     matrix.base_row(cases[1], cfg, "success", start=0, end=0)])
            self.assertEqual(matrix.successful_cases(path, cases, cfg), {c["case_id"] for c in cases[2:]})
            with path.open("a") as stream:
                row = matrix.base_row(cases[2], cfg, "success", start=0, end=1)
                csv.writer(stream).writerow([row[field] for field in matrix.FIELDS[:-1]])
            self.assertEqual(matrix.successful_cases(path, cases, cfg), {cases[3]["case_id"]})

    def test_ctrl_c_collects_checkpoint_and_restart_skips_completed_case(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.config(directory)
            cfg = matrix.load_config(path)
            cases = matrix.make_topology_cases(cfg, 1, 2)
            launcher = Path(directory) / "launcher.sh"
            ready = Path(directory) / "ready"
            launcher.write_text("""#!/bin/bash
set -eu
trap 'cp "$CASE_RESULT_FILE.container" "$CASE_RESULT_FILE"; exit 143' TERM
python3 - <<'PYTHON'
import json, os
from pathlib import Path
cases = json.loads(os.environ['MATRIX_CASES_JSON'])
Path(os.environ['CASE_RESULT_FILE'] + '.container').write_text(json.dumps({
    'results': [dict(cases[0], status='success', group_start=0, group_end=1)],
    'current_case_id': cases[1]['case_id'], 'complete': False}))
Path(os.environ['READY_FILE']).touch()
PYTHON
while true; do sleep 0.1 & wait $!; done
""")
            argv = [str(ROOT / "scripts" / "python" / "run_matrix.py"), "--config", str(path), "--launcher", str(launcher),
                    "--log-dir", str(Path(directory) / "logs")]
            env = dict(os.environ, NPROC_PER_NODE="2", HCCL_NODES="", READY_FILE=str(ready))
            process = subprocess.Popen([sys.executable, *argv], env=env,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            try:
                deadline = time.monotonic() + 5
                while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertTrue(ready.exists())
                process.send_signal(signal.SIGINT)
                output, _ = process.communicate(timeout=10)
                self.assertEqual(process.returncode, 130, output)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate()
            completed = matrix.successful_cases(Path(cfg["output_csv"]), cases, cfg)
            self.assertEqual(completed, {cases[0]["case_id"]})
            with mock.patch("sys.argv", argv + ["--dry-run"]), \
                 mock.patch.dict(os.environ, env), \
                 mock.patch("sys.stdout", new_callable=io.StringIO) as stdout:
                self.assertEqual(matrix.main(), 0)
            self.assertIn("1 successful cases skipped; 3 combinations remaining", stdout.getvalue())

    def test_unimported_checkpoint_recovery_is_idempotent_and_checks_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = matrix.load_config(self.config(directory))
            cases = matrix.make_topology_cases(cfg, 1, 2)
            output = Path(cfg["output_csv"])
            logs = Path(directory) / "logs"
            attempt = logs / "batch_test" / "attempt_1"
            attempt.mkdir(parents=True)
            (attempt / "attempt.json").write_text(json.dumps({
                "output_csv": str(output), "warmup": cfg["warmup"], "repeat": cfg["repeat"],
                "case_ids": [c["case_id"] for c in cases]}))
            (attempt / "checkpoint.json").write_text(json.dumps({"results": [
                dict(cases[0], status="success", group_start=0, group_end=1)],
                "current_case_id": cases[1]["case_id"], "complete": False}))
            recover = lambda settings=cfg, dry=False: matrix.recover_checkpoints(
                logs, output, cases, settings, set(), dry)
            self.assertEqual(recover(dict(cfg, repeat=2)), set())
            self.assertEqual(recover(dry=True), {cases[0]["case_id"]})
            self.assertFalse(output.exists())
            self.assertEqual(recover(), {cases[0]["case_id"]})
            contents = output.read_text()
            self.assertEqual(recover(), set())
            self.assertEqual(output.read_text(), contents)
            matrix.append_rows(output, [matrix.base_row(cases[0], cfg, "failed")])
            self.assertEqual(recover(), set())
            self.assertEqual(matrix.successful_cases(output, cases, cfg), set())

    def config(self, directory):
        path = Path(directory) / "config.json"
        path.write_text(json.dumps({
            "operators": ["allreduce", "broadcast"],
            "sizes_bytes": [16, 32], "dtype": "float16", "group_sizes": [2],
            "execution_modes": ["eager"], "expansion_modes": ["DEFAULT"],
            "warmup": 0, "repeat": 1, "case_timeout_sec": 1,
            "output_csv": str(Path(directory) / "results.csv"),
        }))
        return path

    def test_cases_group_by_communicator_settings_preserving_order(self):
        cases = matrix.make_cases(matrix.load_config(self.config(tempfile.mkdtemp())))
        batches = matrix.group_cases(cases)
        self.assertEqual(len(batches), 1)
        self.assertEqual([(c["operator"], c["size_bytes"]) for c in batches[0][1]],
                         [("allreduce", 16), ("allreduce", 32),
                          ("broadcast", 16), ("broadcast", 32)])

    def test_size_scale_factors_expand_in_order_and_deduplicate(self):
        self.assertEqual(matrix.expand_sizes([16, 32], [2, 4]),
                         [16, 8, 4, 32, 64, 128])
        cfg = matrix.load_config(self.config(tempfile.mkdtemp()))
        cfg["sizes_bytes"] = [16, 32]
        cfg["size_scale_factors"] = [2, 4]
        cases = matrix.make_cases(cfg)
        self.assertEqual(list(dict.fromkeys(c["size_bytes"] for c in cases)),
                         [16, 8, 4, 32, 64, 128])

    def test_size_scale_factors_default_empty_and_validate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.config(directory)
            cfg = matrix.load_config(path)
            self.assertEqual(cfg["size_scale_factors"], [])
            self.assertEqual(matrix.expand_sizes([5, 16], []), [5, 16])
            self.assertEqual(matrix.expand_sizes([5], [2]), [5, 10])
            for factors in ([1], [3], [2.0], [True], "2"):
                invalid = json.loads(path.read_text())
                invalid["size_scale_factors"] = factors
                path.write_text(json.dumps(invalid))
                with self.subTest(factors=factors), self.assertRaisesRegex(
                        ValueError, "size_scale_factors"):
                    matrix.load_config(path)

    def test_size_expansion_caps_sizes_at_eight_gib(self):
        cap = 8 * 1024 ** 3
        self.assertEqual(matrix.expand_sizes([cap // 2], [2, 4]),
                         [cap // 2, cap // 4, cap // 8, cap])
        self.assertEqual(matrix.expand_sizes([cap], [2]), [cap, cap // 2])
        self.assertEqual(matrix.expand_sizes([cap + 1], []), [])

    def test_allgather_limit_applies_to_group_output_size(self):
        case = {"operator": "allgather", "size_bytes": 2 * 1024 ** 3,
                "group_size": 8, "dtype": "float16", "expansion_mode": "DEFAULT"}
        self.assertEqual(matrix.validate_case(case, 8),
                         "allgather output size exceeds 8 GiB limit")
        case["group_size"] = 4
        self.assertEqual(matrix.validate_case(case, 4), "")

    def test_large_case_lists_are_split_under_environment_limit(self):
        cases = [{"case_id": f"case_{index:04d}", "operator": "allreduce",
                  "size_bytes": index + 1} for index in range(20)]
        batches = matrix.split_cases_by_env_size([("key", cases)], max_bytes=500)
        self.assertGreater(len(batches), 1)
        self.assertEqual([case["case_id"] for _, batch in batches for case in batch],
                         [case["case_id"] for case in cases])
        for _, batch in batches:
            payload = json.dumps([{"case_id": case["case_id"],
                                   "operator": case["operator"],
                                   "size_bytes": case["size_bytes"]} for case in batch],
                                 separators=(",", ":"))
            self.assertLessEqual(len(payload.encode("utf-8")), 500)

    def test_dtype_is_required_and_taken_from_json(self):
        with tempfile.TemporaryDirectory() as directory:
            config = json.loads(self.config(directory).read_text())
            self.assertEqual(matrix.make_cases(config)[0]["dtype"], "float16")
            del config["dtype"]
            path = Path(directory) / "missing_dtype.json"
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "missing config key: dtype"):
                matrix.load_config(path)

    def test_case_name_includes_group_and_cluster_topology(self):
        cfg = matrix.load_config(self.config(tempfile.mkdtemp()))
        cfg["group_sizes"] = [8]
        case = matrix.make_cases(cfg, node_count=2, nproc_per_node=4)[0]
        self.assertTrue(case["case_id"].startswith("g8_n2_p4_group8_"), case["case_id"])

    def test_benchmark_uses_world_without_creating_subgroups(self):
        tree = ast.parse((ROOT / "scripts" / "python" / "run_bench.py").read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == "select_world_group")
        dist = mock.Mock()
        dist.get_world_size.return_value = 8
        namespace = {"dist": dist}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "run_bench.py", "exec"), namespace)
        group, start = namespace["select_world_group"](8)
        self.assertIs(group, dist.group.WORLD)
        self.assertEqual(start, 0)
        dist.new_group.assert_not_called()
        with self.assertRaisesRegex(ValueError, "must equal"):
            namespace["select_world_group"](4)

    def test_topology_capacity_and_launch_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.config(directory)
            cfg = matrix.load_config(path)
            cfg["group_sizes"] = [8]
            path.write_text(json.dumps(cfg))
            cases = matrix.make_topology_cases(cfg, 2, 8)
            self.assertEqual([(key[4], key[5]) for key, _ in matrix.group_cases(cases)],
                             [(2, 4), (1, 8)])
            self.assertTrue(all(c["total_ranks"] == c["group_size"] == 8 for c in cases))
            self.assertEqual(len({c["case_id"] for c in cases}), 8)
            limited = matrix.make_topology_cases(cfg, 2, 4)
            self.assertEqual({(c["node_count"], c["nproc_per_node"]) for c in limited}, {(2, 4)})
            impossible = matrix.make_topology_cases(cfg, 1, 4)
            self.assertTrue(all("no topology" in matrix.validate_case(c, c["total_ranks"])
                                for c in impossible))
            launches = []

            def launch(cmd, env, timeout, log_path, label):
                batch = json.loads(env["MATRIX_CASES_JSON"])
                launches.append((cmd[2:], int(env["NPROC_PER_NODE"]), len(batch)))
                self.assertEqual(env["GROUP_SIZE"], "8")
                payload = {"complete": True, "current_case_id": None, "results": [
                    dict(c, group_start=0, group_end=7, status="success") for c in batch]}
                Path(env["CASE_RESULT_FILE"]).write_text(json.dumps(payload))
                return 0

            with mock.patch("sys.argv", ["run_matrix.py", "--config", str(path),
                                         "--nodes", "10.0.0.2", "--log-dir", str(Path(directory) / "logs")]), \
                 mock.patch.dict(os.environ, {"NPROC_PER_NODE": "8"}), \
                 mock.patch.object(matrix, "run_launcher", side_effect=launch):
                self.assertEqual(matrix.main(), 0)
            self.assertEqual(launches, [(["10.0.0.2"], 4, 4), ([], 8, 4)])

    def test_failed_case_checkpoint_resumes_only_remaining_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory)
            cases = matrix.make_cases(matrix.load_config(config), node_count=2, nproc_per_node=1)
            launches = []

            def fake_launcher(cmd, env, timeout, log_path, label):
                batch = json.loads(env["MATRIX_CASES_JSON"])
                launches.append([case["case_id"] for case in batch])
                rows = []
                if len(launches) == 1:
                    rows.append({"case_id": batch[0]["case_id"], "operator": batch[0]["operator"],
                        "size_bytes": batch[0]["size_bytes"], "group_start": 0, "group_end": 1,
                        "input_bytes": batch[0]["size_bytes"], "output_bytes": batch[0]["size_bytes"],
                        "metrics": {"mean_ms": 1.0}, "status": "success"})
                    payload = {"results": rows, "current_case_id": batch[1]["case_id"], "complete": False}
                    code = 1
                else:
                    for item in batch:
                        rows.append({"case_id": item["case_id"], "operator": item["operator"],
                            "size_bytes": item["size_bytes"], "group_start": 0, "group_end": 1,
                            "input_bytes": item["size_bytes"], "output_bytes": item["size_bytes"],
                            "metrics": {"mean_ms": 2.0}, "status": "success"})
                    payload = {"results": rows, "current_case_id": None, "complete": True}
                    code = 0
                Path(env["CASE_RESULT_FILE"]).write_text(json.dumps(payload))
                return code

            with mock.patch("sys.argv", ["run_matrix.py", "--config", str(config),
                                           "--nodes", "127.0.0.2", "--log-dir", str(Path(directory) / "logs")]), \
                 mock.patch.dict(os.environ, {"NPROC_PER_NODE": "1"}), \
                 mock.patch.object(matrix, "run_launcher", side_effect=fake_launcher):
                self.assertEqual(matrix.main(), 0)
            self.assertEqual(len(launches), 2)
            self.assertEqual(launches[0], [case["case_id"] for case in cases])
            self.assertEqual(launches[1], [cases[2]["case_id"], cases[3]["case_id"]])
            with Path(matrix.load_config(config)["output_csv"]).open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["case_id"] for row in rows], [case["case_id"] for case in cases])
            self.assertEqual([row["status"] for row in rows], ["success", "failed", "success", "success"])


if __name__ == "__main__":
    unittest.main()
