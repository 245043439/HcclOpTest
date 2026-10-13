"""CPU-only checks for matrix batching and checkpoint recovery."""
import csv
import ast
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("run_matrix", ROOT / "run_matrix.py")
matrix = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(matrix)


class BatchingTests(unittest.TestCase):
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
        tree = ast.parse((ROOT / "run_bench.py").read_text())
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
