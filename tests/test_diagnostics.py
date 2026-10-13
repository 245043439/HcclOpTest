"""Exercise failure logging without torch, Docker or an NPU."""
import os
import ast
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class DiagnosticsTests(unittest.TestCase):
    def test_cleanup_collects_checkpoint_even_without_run_worker(self):
        source = (ROOT / "launch_cluster.sh").read_text()
        cleanup = source.split("\ncleanup() {", 1)[1].split("\non_exit()", 1)[0]
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "cleanup.sh"
            script.write_text("""set -eu
LOG_DIR="$1"
CASE_RESULT_FILE="$1/checkpoint.json"
REMOTE_RESULT_FILE=/tmp/container-result.json
CONTAINER_NAME=test
WORKER_SCRIPT="$1/worker"
CLEANING=0
CLEANUP_STATUS=0
WATCHDOG_PID=''
PREPARED=1
PIDS=()
NODES=(master)
invoke() { return 0; }
docker() {
    [[ "$1" == cp && "$2" == test:/tmp/container-result.json ]]
    echo '{"results": []}' > "$3"
}
cleanup() {""" + cleanup + "\ncleanup\n")
            result = subprocess.run(["bash", str(script), directory], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((Path(directory) / "checkpoint.json").read_text(), '{"results": []}\n')
            self.assertFalse((Path(directory) / "checkpoint.json.tmp").exists())

    def test_stale_process_check_ignores_empty_cmdline(self):
        source = (ROOT / 'launch_cluster.sh').read_text()
        worker = source.split("<<'REMOTE_SCRIPT'\n", 1)[1].split('\nREMOTE_SCRIPT\n', 1)[0]
        checker = worker.split("docker exec -i \"$CONTAINER\" python3 - \"$BENCH_SCRIPT\" <<'PY'\n", 1)[1].split('\nPY\n', 1)[0]
        tree = ast.parse(checker)
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == 'is_benchmark_process')
        namespace = {'os': os}
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'check_stale', 'exec'), namespace)
        self.assertFalse(namespace['is_benchmark_process']([], 'run_bench.py'))
        self.assertTrue(namespace['is_benchmark_process'](['python3', 'run_bench.py'], 'run_bench.py'))

    def test_launcher_watchdog_tracks_sleep_process_directly(self):
        source = (ROOT / 'launch_cluster.sh').read_text()
        self.assertIn('sleep "$TIMEOUT_SEC" &\nWATCHDOG_PID=$!', source)
        self.assertNotIn('( sleep "$TIMEOUT_SEC"; exit 124 ) &', source)

    def test_latency_max_reduction_uses_hccl_supported_float32(self):
        source = (ROOT / 'run_bench.py').read_text()
        tree = ast.parse(source)
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == 'max_latency')

        class Tensor:
            def item(self):
                return 1.25

        class FakeTorch:
            float32 = object()
            observed_dtype = None

            @classmethod
            def tensor(cls, values, **kwargs):
                cls.observed_dtype = kwargs['dtype']
                return Tensor()

        class FakeDist:
            class ReduceOp:
                MAX = object()

            @staticmethod
            def get_rank():
                return 0

            @staticmethod
            def all_reduce(*args, **kwargs):
                return None

        namespace = {'torch': FakeTorch, 'dist': FakeDist}
        exec(compile(ast.Module(body=[function], type_ignores=[]),
                     str(ROOT / 'run_bench.py'), 'exec'), namespace)
        result = namespace['max_latency'](1.25, object())
        self.assertEqual(result, 1.25)
        self.assertIs(FakeTorch.observed_dtype, FakeTorch.float32)

    def test_nonleader_initialization_exception_keeps_chained_traceback(self):
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            (temp / 'torch').mkdir()
            (temp / 'torch' / '__init__.py').write_text('''
float16 = object()
float32 = object()
bfloat16 = object()
int8 = object()
int16 = object()
int32 = object()
int64 = object()
uint8 = object()
class NPU:
    def set_device(self, device):
        try:
            raise ValueError("device driver failure")
        except ValueError as exc:
            raise RuntimeError("cannot initialize device") from exc
npu = NPU()
''')
            (temp / 'torch' / 'distributed.py').write_text('')
            (temp / 'torch_npu.py').write_text('')
            env = dict(os.environ, PYTHONPATH=str(temp), RANK='3', LOCAL_RANK='1',
                       HCCL_BENCH_DIAG_DIR=str(temp / 'diagnostics'),
                       HCCL_BENCH_STACK_INTERVAL='0', PYTHONDONTWRITEBYTECODE='1')
            result = subprocess.run(['python3', str(ROOT / 'run_bench.py'),
                                     '--size-bytes', '16', '--group-size', '2'],
                                    env=env, capture_output=True, text=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            rank_log = (temp / 'diagnostics' / 'rank_3.traceback.log').read_text()
            for log in (result.stderr, rank_log):
                self.assertIn('Traceback (most recent call last)', log)
                self.assertIn('ValueError: device driver failure', log)
                self.assertIn('RuntimeError: cannot initialize device', log)
                self.assertIn('rank=3 local_rank=1 stage=set NPU device', log)
                self.assertIn('set_device', log)

    def test_worker_failure_preserves_exit_and_shell_location(self):
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            source = (ROOT / 'launch_cluster.sh').read_text()
            worker = source.split("<<'REMOTE_SCRIPT'\n", 1)[1].split('\nREMOTE_SCRIPT\n', 1)[0]
            (temp / 'worker.sh').write_text(worker)
            docker = temp / 'docker'
            docker.write_text('#!/bin/bash\necho "injected docker failure" >&2\nexit 23\n')
            docker.chmod(0o755)
            args = ['run', '1', '2', '127.0.0.1', '29600', 'test-container', 'test-image',
                    'docker-gen.sh', 'run_bench.py', '8', '1', '2', '0', '1', 'test-run',
                    'allreduce', 'eager', 'DEFAULT', '1024', '/tmp/test-result.json', 'float32']
            env = dict(os.environ, PATH=str(temp) + os.pathsep + os.environ['PATH'])
            result = subprocess.run(['bash', str(temp / 'worker.sh'), *args], env=env,
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 23)
            self.assertIn('injected docker failure', result.stderr)
            self.assertIn('[Node 1] mode=run failed exit=23 line=', result.stderr)
            self.assertIn('command=docker exec', result.stderr)
            self.assertIn('worker.sh', result.stderr)


if __name__ == '__main__':
    unittest.main()
