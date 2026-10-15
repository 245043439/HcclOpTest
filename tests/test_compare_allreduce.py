"""CPU-only tests of full communication filtering and comparison semantics."""
import contextlib
import csv
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "compare_allreduce", ROOT / "scripts/python/compare_allreduce.py")
compare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare)


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def export(self, directory, durations=(999, 10, 20), unit="us", extra=()):
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "op_summary_0.csv"
        scale = {"us": 1, "ms": 1000, "ns": 0.001}[unit]
        headers = ["Op Name", "Op Type", "Task Type", "Stream ID", "Task ID",
                   f"Task Start Time({unit})", f"Task Duration({unit})"]
        rows = [[f"hcom_allReduce_{i}", "AllReduce", "COMMUNICATION", "N/A", "N/A",
                 (i + 1) * 10000 / scale, duration / scale]
                for i, duration in enumerate(durations)]
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(headers)
            writer.writerows(list(reversed(rows)) + list(extra))
        return path

    def raw(self, directory, rank, samples, profiled):
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"rank_{rank}.json"
        path.write_text(json.dumps({
            "schema_version": 1, "rank": rank, "local_rank": rank, "world_size": 2,
            "size_bytes": 1024, "dtype": "float32", "warmup": 1, "repeat": 2,
            "execution_mode": "eager", "expansion_mode": "DEFAULT",
            "profiled": profiled, "samples_us": samples, "metadata": {"hostname": "worker"}}))
        return path

    def setup_runs(self):
        base, prof = self.root / "baseline", self.root / "profiled"
        self.raw(base, 0, [100, 300], False)
        self.raw(base, 1, [200, 150], False)
        self.raw(prof, 0, [150, 400], True)
        self.raw(prof, 1, [250, 180], True)
        self.export(prof / "profiler/rank_0", (999, 110, 120))
        self.export(prof / "profiler/rank_1", (999, 100, 140))
        return base, prof

    def test_units_warmup_order_and_auxiliary_filtering(self):
        for unit in ("us", "ms", "ns"):
            with self.subTest(unit=unit):
                directory = self.root / unit
                self.export(directory, unit=unit, extra=[
                    ["hcom_barrier", "Barrier", "COMMUNICATION", "N/A", "N/A", 1, 100],
                    ["hcom_allReduce_kernel", "AllReduce", "COMMUNICATION", 3, 4, 2, 100],
                    ["hcom_allReduce_api", "AllReduce", "AI_CPU", "N/A", "N/A", 3, 100]])
                events = compare.read_communication(directory, 1, 2)
                self.assertEqual([e["duration_us"] for e in events], [10, 20])
                self.assertEqual([e["start_us"] for e in events], [20000, 30000])

    def test_full_analysis_maxima_and_paired_differences(self):
        base, prof = self.setup_runs()
        output = self.root / "report"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(compare.main(["analyze", "--baseline", str(base),
                                           "--profiled", str(prof), "--output", str(output)]), 0)
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(summary["metrics"]["baseline_e2e_us"]["mean"], 250)
        self.assertEqual(summary["metrics"]["profiled_e2e_us"]["mean"], 325)
        self.assertEqual(summary["metrics"]["communication_us"]["mean"], 125)
        self.assertAlmostEqual(summary["ratios_of_means_pct"]["profiler_change"], 30)
        with (output / "iterations.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(rows[0]["profiled_max_rank"], "1")
        self.assertEqual(rows[0]["communication_max_rank"], "0")
        self.assertEqual(float(rows[0]["e2e_minus_communication_us"]), 140)
        with (output / "rank_pairs.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(float(rows[0]["e2e_minus_communication_us"]), 40)
        self.assertEqual(float(rows[1]["e2e_minus_communication_us"]), 150)

    def test_missing_extra_and_duplicate_events_fail(self):
        for durations in ((1, 2), (1, 2, 3, 4)):
            self.export(self.root, durations)
            with self.assertRaisesRegex(ValueError, "expected 3"):
                compare.read_communication(self.root, 1, 2)
        path = self.export(self.root)
        (self.root / "op_summary_duplicate.csv").write_text(path.read_text())
        with self.assertRaisesRegex(ValueError, "expected 3"):
            compare.read_communication(self.root, 1, 2)

    def test_missing_rank_and_sample_count_fail(self):
        self.raw(self.root, 0, [1, 2], False)
        with self.assertRaisesRegex(ValueError, "missing or unexpected ranks"):
            compare.load_run(self.root, False)
        self.raw(self.root, 1, [1], False)
        with self.assertRaisesRegex(ValueError, "sample count"):
            compare.load_run(self.root, False)

    def test_invalid_durations_fail(self):
        for value in (0, -1, float("nan"), float("inf")):
            self.export(self.root, (1, value, 3))
            with self.assertRaises(ValueError):
                compare.read_communication(self.root, 1, 2)

    def test_unsupported_schema_and_kernel_only_fail(self):
        path = self.export(self.root)
        path.write_text(path.read_text().replace("Task Duration(us)", "Duration"))
        with self.assertRaisesRegex(ValueError, "unsupported"):
            compare.read_communication(self.root, 1, 2)
        path = self.export(self.root)
        path.write_text(path.read_text().replace("N/A,N/A", "1,2"))
        with self.assertRaisesRegex(ValueError, "found 0"):
            compare.read_communication(self.root, 1, 2)

    def test_mismatched_config_fails_before_writing_report(self):
        base, prof = self.setup_runs()
        for rank in range(2):
            path = prof / f"rank_{rank}.json"
            data = json.loads(path.read_text())
            data["size_bytes"] = 2048
            path.write_text(json.dumps(data))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
            compare.main(["analyze", "--baseline", str(base), "--profiled", str(prof),
                          "--output", str(self.root / "report")])
        self.assertEqual(exc.exception.code, 2)
        self.assertFalse((self.root / "report").exists())

    def test_percentile_matches_existing_benchmark(self):
        values = list(range(1, 51))
        self.assertEqual(compare.stats(values)["p95"], 47)
        self.assertEqual(compare.stats(values)["p50"], 25.5)

    def test_run_preserves_timing_order_without_extra_allreduce(self):
        calls = []
        torch = types.ModuleType("torch")
        torch.__version__ = "test"
        torch.float32 = "float32"
        torch.zeros = mock.Mock(return_value="tensor")
        torch.npu = types.SimpleNamespace(
            set_device=lambda rank: None,
            synchronize=lambda: calls.append("sync"),
            get_device_properties=lambda rank: types.SimpleNamespace(name="test"))
        dist = types.ModuleType("torch.distributed")
        dist.init_process_group = lambda **kwargs: None
        dist.destroy_process_group = lambda: calls.append("destroy")
        dist.barrier = lambda: calls.append("barrier")
        dist.all_reduce = lambda tensor: calls.append("allreduce")
        torch.distributed = dist
        npu = types.ModuleType("torch_npu")
        clock = iter([1, 1.0001, 2, 2.0002])

        def timestamp():
            calls.append("time")
            return next(clock)

        with mock.patch.dict("sys.modules", {"torch": torch, "torch_npu": npu,
                                             "torch.distributed": dist}), \
                mock.patch.dict("os.environ", {"RANK": "0", "WORLD_SIZE": "2", "LOCAL_RANK": "0"}), \
                mock.patch.object(compare.time, "perf_counter", side_effect=timestamp), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(compare.main(["run", "--output", str(self.root),
                                           "--warmup", "1", "--repeat", "2"]), 0)
        self.assertEqual(calls, ["allreduce", "sync", "barrier"] +
                         ["barrier", "sync", "time", "allreduce", "sync", "time"] * 2 + ["destroy"])
        result = json.loads((self.root / "rank_0.json").read_text())
        self.assertAlmostEqual(result["samples_us"][0], 100)
        self.assertAlmostEqual(result["samples_us"][1], 200)
        self.assertFalse(result["profiled"])
        torch.zeros.assert_called_once_with(262144, dtype="float32", device="npu:0")

    def test_msprof_wraps_each_rank_and_propagates_exit(self):
        with mock.patch.dict("os.environ", {"RANK": "1", "WORLD_SIZE": "2", "LOCAL_RANK": "1"}), \
                mock.patch.object(compare.subprocess, "run", return_value=types.SimpleNamespace(returncode=7)) as launch:
            self.assertEqual(compare.main(["run", "--msprof", "--output", str(self.root)]), 7)
        command = launch.call_args.args[0]
        self.assertEqual(command[0], "msprof")
        self.assertIn(f"--output={self.root}/profiler/rank_1", command)
        self.assertIn("--profiled", command)
        self.assertNotIn("--msprof", command)


if __name__ == "__main__":
    unittest.main()
