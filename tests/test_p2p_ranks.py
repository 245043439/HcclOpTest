"""Physical endpoint reporting must not change communicator ranks or resume."""
import csv
import json
import importlib.util
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("run_matrix", Path(__file__).resolve().parents[1] / "run_matrix.py")
matrix = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(matrix)


class EndpointTests(unittest.TestCase):
    def case(self, nodes, procs, op="send_recv"):
        return dict(case_id=f"g{nodes * procs}_n{nodes}_p{procs}_group{nodes * procs}_test", operator=op, node_count=nodes, nproc_per_node=procs,
                    total_ranks=nodes * procs, group_size=nodes * procs, size_bytes=16,
                    dtype="float16", execution_mode="eager", expansion_mode="DEFAULT")

    def test_cross_node_and_local_pairs(self):
        cfg = dict(warmup=0, repeat=1, rank_stride=8)
        for nodes, procs, target in [(2, 1, [0, 8]), (1, 2, [0, 1]), (2, 2, [0, 1, 8, 9])]:
            with self.subTest(nodes=nodes, procs=procs):
                case = self.case(nodes, procs)
                row = matrix.result_row(case, cfg, dict(status="success", group_start=0,
                                        group_end=nodes * procs - 1), {}, "/tmp/log")
                self.assertEqual(json.loads(row["rank_list"]), target)
                self.assertNotIn("rank_end", row)
                self.assertNotIn("source_rank", row)
        self.assertEqual(matrix.global_ranks(self.case(2, 1), {"rank_stride": 16}), [0, 16])

    def test_ring_edges(self):
        for op in matrix.OPS:
            self.assertEqual(matrix.global_ranks(self.case(2, 2, op), {}), [0, 1, 8, 9])
        with self.assertRaises(ValueError):
            matrix.global_ranks(self.case(1, 4), {"rank_stride": 2})

    def test_legacy_csv_upgrade_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            case = self.case(2, 1)
            cfg = dict(warmup=0, repeat=1)
            row = matrix.base_row(case, cfg, "success", start=0, end=1)
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=matrix.LEGACY_FIELDS, extrasaction="ignore")
                writer.writeheader()
                writer.writerow(dict(row, rank_start=0, rank_end=1))
            self.assertEqual(matrix.successful_cases(path, [case], cfg), {case["case_id"]})
            matrix.append_rows(path, [row])
            with path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertEqual(json.loads(rows[0]["rank_list"]), [0, 8])
            self.assertEqual(json.loads(rows[1]["rank_list"]), [0, 8])
            self.assertEqual(matrix.successful_cases(path, [case], cfg), {case["case_id"]})

    def test_partial_or_wrong_rank_list_does_not_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            case = self.case(2, 2, "allreduce")
            cfg = dict(warmup=0, repeat=1)
            row = matrix.base_row(case, cfg, "success", start=0, end=1)
            matrix.append_rows(path, [row])
            self.assertEqual(matrix.successful_cases(path, [case], cfg), set())
            matrix.append_rows(path, [matrix.base_row(case, cfg, "success")])
            self.assertEqual(matrix.successful_cases(path, [case], cfg), {case["case_id"]})
            self.assertEqual(matrix.successful_cases(path, [case], dict(cfg, rank_stride=16)), set())


if __name__ == "__main__":
    unittest.main()
