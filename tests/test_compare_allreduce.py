"""CPU-only tests for the eager AllReduce launcher."""
import contextlib
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


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

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
