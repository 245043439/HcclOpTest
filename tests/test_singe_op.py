"""CPU-only tests for the single-operator launcher and graph execution."""
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
    "singe_op", ROOT / "scripts/python/singe_op.py")
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_run_preserves_timing_order_without_extra_allreduce(self):
        self.check_run("eager")

    def test_aclgraph_preparation_is_outside_timing_and_iterations_replay(self):
        self.check_run("aclgraph")

    def test_ascend_ir_compilation_is_outside_timing(self):
        self.check_run("ascend_ir")

    def test_alltoall_execution_modes(self):
        for mode in ("eager", "aclgraph", "ascend_ir"):
            with self.subTest(mode=mode):
                self.check_run(mode, "alltoall")
                (self.root / "rank_0.json").unlink()

    def check_run(self, mode, operator="allreduce"):
        calls = []
        torch = types.ModuleType("torch")
        torch.__version__ = "test"
        torch.float32 = "float32"
        torch.zeros = mock.Mock(return_value="tensor")
        torch.empty_like = mock.Mock(return_value="output_tensor")
        torch.npu = types.SimpleNamespace(
            set_device=lambda rank: None,
            synchronize=lambda: calls.append("sync"),
            get_device_properties=lambda rank: types.SimpleNamespace(name="test"))
        dist = types.ModuleType("torch.distributed")
        dist.init_process_group = lambda **kwargs: None
        dist.destroy_process_group = lambda: calls.append("destroy")
        dist.barrier = lambda: calls.append("barrier")
        dist.all_reduce = lambda tensor: calls.append("allreduce")
        dist.all_to_all_single = mock.Mock(side_effect=lambda out, x: calls.append("alltoall"))
        torch.distributed = dist
        npu = types.ModuleType("torch_npu")
        torchair = types.ModuleType("torchair")
        torchair.patch_for_hcom = lambda: calls.append("patch")
        torchair.get_npu_backend = lambda: "backend"
        compiled = lambda: calls.append("compiled")
        torch.compile = mock.Mock(return_value=compiled)
        graph = types.SimpleNamespace(replay=lambda: calls.append("replay"))
        torch.npu.NPUGraph = lambda: graph

        @contextlib.contextmanager
        def capture(value):
            self.assertIs(value, graph)
            calls.append("capture_start")
            yield
            calls.append("capture_end")

        torch.npu.graph = capture
        clock = iter([1, 1.0001, 2, 2.0002])

        def timestamp():
            calls.append("time")
            return next(clock)

        with mock.patch.dict("sys.modules", {"torch": torch, "torch_npu": npu,
                                             "torch.distributed": dist, "torchair": torchair}), \
                mock.patch.dict("os.environ", {"RANK": "0", "WORLD_SIZE": "2", "LOCAL_RANK": "0"}), \
                mock.patch.object(bench.time, "perf_counter", side_effect=timestamp), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(bench.main(["run", "--output", str(self.root),
                                           "--execution-mode", mode,
                                           "--operator", operator,
                                           "--warmup", "1", "--repeat", "2"]), 0)
        preparation = {
            "eager": [],
            "aclgraph": [operator, "sync", "capture_start", operator, "capture_end", "sync"],
            "ascend_ir": ["patch", "compiled", "sync"],
        }[mode]
        operation = {"eager": operator, "aclgraph": "replay", "ascend_ir": "compiled"}[mode]
        self.assertEqual(calls, preparation + [operation, "sync", "barrier"] +
                         ["barrier", "sync", "time", operation, "sync", "time"] * 2 + ["destroy"])
        if mode == "ascend_ir":
            self.assertEqual(torch.compile.call_args.kwargs, {"backend": "backend", "fullgraph": True})
        result = json.loads((self.root / "rank_0.json").read_text())
        self.assertEqual(result["execution_mode"], mode)
        self.assertEqual(result["operator"], operator)
        self.assertAlmostEqual(result["samples_us"][0], 100)
        self.assertAlmostEqual(result["samples_us"][1], 200)
        self.assertFalse(result["profiled"])
        torch.zeros.assert_called_once_with(262144, dtype="float32", device="npu:0")
        if operator == "alltoall":
            torch.empty_like.assert_called_once_with("tensor")
            if mode == "ascend_ir":
                # Inspect the callable passed to the compiler without needing an NPU.
                torch.compile.call_args.args[0]()
            self.assertTrue(dist.all_to_all_single.called)
            for call in dist.all_to_all_single.call_args_list:
                self.assertEqual(call.args, ("output_tensor", "tensor"))

    def test_msprof_wraps_each_rank_and_propagates_exit(self):
        with mock.patch.dict("os.environ", {"RANK": "1", "WORLD_SIZE": "2", "LOCAL_RANK": "1"}), \
                mock.patch.object(bench.subprocess, "run", return_value=types.SimpleNamespace(returncode=7)) as launch:
            self.assertEqual(bench.main(["run", "--msprof", "--output", str(self.root),
                                           "--execution-mode", "aclgraph", "--operator", "alltoall"]), 7)
        command = launch.call_args.args[0]
        self.assertEqual(command[0], "msprof")
        self.assertIn(f"--output={self.root}/profiler/rank_1", command)
        self.assertIn("--profiled", command)
        self.assertNotIn("--msprof", command)
        self.assertEqual(command[command.index("--execution-mode") + 1], "aclgraph")
        self.assertEqual(command[command.index("--operator") + 1], "alltoall")
        self.assertIn(str(ROOT / "scripts/python/singe_op.py"), command)

    def test_alltoall_rejects_uneven_split_before_launch(self):
        with mock.patch.dict("os.environ", {"RANK": "0", "WORLD_SIZE": "2", "LOCAL_RANK": "0"}), \
                mock.patch.object(bench.subprocess, "run") as launch, \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
            bench.main(["run", "--msprof", "--operator", "alltoall", "--size-bytes", "12",
                        "--output", str(self.root / "invalid")])
        self.assertEqual(exc.exception.code, 2)
        launch.assert_not_called()
        self.assertFalse((self.root / "invalid").exists())

    def test_missing_graph_apis_fail_without_eager_fallback(self):
        fn = mock.Mock()
        torch = types.SimpleNamespace(npu=types.SimpleNamespace())
        with self.assertRaisesRegex(RuntimeError, "capture API is unavailable"):
            bench.graph_callable(fn, "aclgraph", torch)
        with mock.patch.dict("sys.modules", {"torchair": types.ModuleType("torchair")}), \
                self.assertRaisesRegex(RuntimeError, "patch_for_hcom API is unavailable"):
            bench.graph_callable(fn, "ascend_ir", torch)
        fn.assert_not_called()


if __name__ == "__main__":
    unittest.main()
