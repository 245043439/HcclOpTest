"""CPU-only regressions for the empirical launch-span model."""

import importlib.util
import itertools
import json
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location(
    "node_sync_model", Path(__file__).resolve().parents[1]
    / "node_sync_model/node_sync_model.py")
model = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(model)


class NodeSyncModelTests(unittest.TestCase):
    def test_integral_matches_exhaustive_paired_sampling(self):
        pairs = [(-3, -1), (0, 4), (0, 0), (2, 4)]
        samples = [(str(i), "0", a, b) for i, (a, b) in enumerate(pairs)]
        cdf = model.empirical_cdf(samples)
        for n in range(1, 5):
            draws = list(itertools.product(pairs, repeat=n))
            expected = sum(max(b for _, b in draw) - min(a for a, _ in draw)
                           for draw in draws) / len(draws)
            self.assertAlmostEqual(model.expected_span(cdf, n), expected)

    def test_constant_intervals_and_large_scale(self):
        for a, b in [(2, 2), (2, 7), (-7, -2)]:
            cdf = model.empirical_cdf([("1", "0", a, b)])
            for n in [1, 2, 1000000]:
                self.assertEqual(model.expected_span(cdf, n), b - a)
        cdf = model.empirical_cdf([("1", "0", 0, 0), ("2", "0", 10, 10)])
        self.assertEqual(model.expected_span(cdf, 1000000), 10)

    def test_invalid_csv(self):
        cases = ["1,0,3,2\n", "1,0,nan,2\n", "1,0,0,inf\n",
                 "1,0,0,2\n1,0,1,3\n", "1,,0,2\n", ""]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.csv"
            for body in cases:
                with self.subTest(body=body):
                    path.write_text("iteration,node_id,A_us,B_us\n" + body)
                    with self.assertRaises(ValueError):
                        model.read_samples(path)

    def test_validation_requires_complete_roster(self):
        samples = [("1", "a", 1, 4), ("1", "b", 2, 8),
                   ("2", "a", 3, 5), ("2", "b", 4, 6)]
        self.assertEqual(model.observed_span(samples), (2, 2, 5))
        with self.assertRaises(ValueError):
            model.observed_span(samples[:-1])

    def test_cli_outputs_and_zero_observed_mean(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, valid = root / "train.csv", root / "valid.csv"
            train.write_text("iteration,node_id,A_us,B_us\n1,0,0,2\n2,0,2,4\n")
            valid.write_text("iteration,node_id,A_us,B_us\n1,0,3,3\n1,1,3,3\n")
            output = root / "output"
            self.assertEqual(model.main([
                "--input", str(train), "--nodes", "1", "2", "4",
                "--validation", str(valid), "--output-dir", str(output)]), 0)
            report = json.loads((output / "report.json").read_text())
            self.assertEqual([r["expected_sync_us"] for r in report["predictions"]],
                             [2, 3, 3.75])
            self.assertIsNone(report["validation"][0]["relative_error"])
            self.assertTrue((output / "cdf.csv").exists())
            self.assertTrue((output / "node_summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
