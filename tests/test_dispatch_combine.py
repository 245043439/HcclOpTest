"""CPU-only coverage of DeepEP benchmark control flow and CSV compatibility."""
import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
with mock.patch.object(sys, "path", [str(ROOT / "scripts/python"), *sys.path]):
    SPEC = importlib.util.spec_from_file_location(
        "dispatch_combine", ROOT / "scripts/python/run_dispatch_combine_bench.py")
    bench = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(bench)


class DispatchCombineTests(unittest.TestCase):
    def setUp(self):
        self.args = bench.parser().parse_args([])
        self.env = {"RANK": "0", "WORLD_SIZE": "4", "LOCAL_RANK": "0",
                    "LOCAL_WORLD_SIZE": "2"}

    def test_validation(self):
        self.assertEqual(bench.validate(self.args, self.env), (0, 4, 0, 2))
        for key, value in (("num_tokens", 0), ("hidden", -1), ("num_topk", 257),
                           ("num_experts", 255), ("repeat", 0), ("warmup", -1),
                           ("rank_stride", 1), ("seed", -1)):
            with self.subTest(key=key), mock.patch.object(self.args, key, value):
                with self.assertRaises(ValueError):
                    bench.validate(self.args, self.env)
        with self.assertRaisesRegex(ValueError, "torchrun"):
            bench.validate(self.args, {})
        for key, value in (("WORLD_SIZE", "3"), ("LOCAL_RANK", "1"),
                           ("DEEP_NORMAL_MODE_USE_INT8_QUANT", "1"),
                           ("MOE_SHARED_EXPERT_RANK_NUM", "1"),
                           ("MOE_ENABLE_TOPK_NEG_ONE", "1"),
                           ("DEEP_USE_MODE", "alltoall"), ("DEEP_USE_MODE", "ops")):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                bench.validate(self.args, {**self.env, key: value})
        self.args.low_latency_strategy = "ops"
        bench.validate(self.args, {**self.env, "DEEP_USE_MODE": "ops"})

    def test_csv_topology_bytes_and_append(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "results.csv"
            for operator in bench.OPERATORS:
                row = bench.result_row(self.args, operator, 4, 2, 123456, {},
                                       bench.statistics_ms([1, 2, 3]))
                bench.append_row(path, row)
            with path.open(newline="") as stream:
                reader = csv.DictReader(stream)
                rows = list(reader)
                self.assertEqual(reader.fieldnames, bench.FIELDS)
            self.assertEqual(len(rows), 3)
            self.assertEqual(json.loads(rows[0]["rank_list"]), [0, 1, 8, 9])
            self.assertEqual(rows[0]["comm_layer_list"], "(1,2,2)")
            self.assertEqual(rows[0]["output_bytes"], "123456")
            self.assertEqual(rows[1]["input_bytes"], "123456")
            self.assertEqual(rows[2]["input_bytes"], rows[2]["output_bytes"])
            self.assertIn("_t256_h7168_k8_e256_bf16_eager_default_seed1_check0_stride8",
                          rows[0]["case_id"])
            old = path.read_text()
            path.write_text("wrong,header\n")
            with self.assertRaisesRegex(ValueError, "header"):
                bench.append_row(path, row)
            self.assertEqual(path.read_text(), "wrong,header\n")
            path.write_text(old)
            bench.append_row(path, row)
            with path.open(newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 4)

    def test_statistics_match_existing_percentile_convention(self):
        values = bench.statistics_ms([4, 1, 3, 2])
        self.assertEqual(values, {"mean_ms": 2.5, "min_ms": 1, "max_ms": 4,
                                  "p50_ms": 2.5, "p95_ms": 3})
        self.assertEqual(bench.statistics_ms([7])["p95_ms"], 7)

    def test_pairing_and_timing_boundaries(self):
        expected = {
            "dispatch": ["barrier", "sync", "clock", "dispatch", "sync", "clock",
                         "combine", "sync"],
            "combine": ["dispatch", "barrier", "sync", "clock", "combine", "sync", "clock"],
            "dispatch_combine": ["barrier", "sync", "clock", "dispatch", "combine", "sync", "clock"],
        }
        for operator, sequence in expected.items():
            with self.subTest(operator=operator):
                calls, handles, used = [], [], []
                ticks = iter([0, 100, 200, 200.001, 300, 300.003])

                def clock():
                    calls.append("clock")
                    return next(ticks)

                def dispatch():
                    calls.append("dispatch")
                    handle = object()
                    handles.append(handle)
                    return handle

                def combine(handle):
                    calls.append("combine")
                    self.assertIs(handle, handles[-1])
                    self.assertNotIn(handle, used)
                    used.append(handle)

                def maximum(value):
                    calls.append("max")
                    return value + 10  # A slower remote rank determines the sample.

                with mock.patch.object(bench.time, "perf_counter", side_effect=clock):
                    metrics = bench.measure(operator, dispatch, combine,
                                            lambda: calls.append("sync"),
                                            lambda: calls.append("barrier"), maximum, 1, 2)
                self.assertEqual(calls, sequence + sequence + ["max"] + sequence + ["max"])
                self.assertEqual(len(used), 3)
                self.assertAlmostEqual(metrics["mean_ms"], 12)
                self.assertAlmostEqual(metrics["min_ms"], 11)
                self.assertAlmostEqual(metrics["max_ms"], 13)


if __name__ == "__main__":
    unittest.main()
