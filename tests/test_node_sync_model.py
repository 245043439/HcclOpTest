"""CPU-only regressions for the empirical launch-span model."""

import importlib.util
import contextlib
import csv
import io
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
    def test_error_averages_per_step_spans_for_required_nodes(self):
        samples = [("1", "0", 0, 2), ("1", "1", 4, 8), ("1", "2", -100, 100),
                   ("2", "0", 100, 102), ("2", "1", 101, 104)]
        result = model.prediction_with_error(samples, 2, 9)
        self.assertEqual(result["y_true_mean_us"], 6)
        self.assertEqual(result["error_us"], 3)
        self.assertEqual(result["absolute_error_us"], 3)
        self.assertEqual(result["relative_error"], 0.5)
        self.assertEqual(result["iteration_count"], 2)
        self.assertEqual(result["error_status"], "ok")
        for n in [3, 4]:
            result = model.prediction_with_error(samples, n, 9)
            self.assertEqual(result["error_status"], "missing_required_nodes")
            self.assertIsNone(result["y_true_mean_us"])
        result = model.prediction_with_error([("1", "0", 2, 2)], 1, 1)
        self.assertEqual(result["error_us"], 1)
        self.assertIsNone(result["relative_error"])

    def test_cli_error_uses_unfiltered_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.csv"
            train.write_text("iteration,node_id,A_us,B_us\n"
                             "1,0,0,2\n1,1,4,8\n2,0,2,4\n2,1,3,6\n")
            output = root / "output"
            with contextlib.redirect_stdout(io.StringIO()) as stdout, contextlib.redirect_stderr(io.StringIO()):
                model.main(["--input", str(train), "--node-id", "0", "--nodes", "2", "4",
                            "--output-dir", str(output)])
            report = json.loads((output / "report.json").read_text())
            row = report["predictions"][0]
            self.assertEqual(row["expected_sync_us"], 3)
            self.assertEqual(row["y_true_mean_us"], 6)
            self.assertEqual(row["error_us"], -3)
            self.assertEqual(row["absolute_error_us"], 3)
            self.assertEqual(row["relative_error"], -0.5)
            with (output / "predictions.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows, list(csv.DictReader(io.StringIO(stdout.getvalue()))))
            self.assertEqual(rows[1]["error_status"], "missing_required_nodes")
            self.assertEqual(rows[1]["error_us"], "")

    def test_node_filter_changes_fit_but_preserves_validation_nodes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, valid = root / "train.csv", root / "valid.csv"
            train.write_text("iteration,node_id,A_us,B_us\n"
                             "1,01,0,2\n2,01,2,4\n1,1,100,200\n")
            valid.write_text("iteration,node_id,A_us,B_us\n1,01,0,2\n1,1,10,20\n")
            for node_id, expected, count in [(None, 104 / 3, 3), ("01", 2, 2)]:
                output = root / str(node_id)
                argv = ["--input", str(train), "--nodes", "1", "2",
                        "--validation", str(valid), "--output-dir", str(output)]
                if node_id is not None:
                    argv.extend(["--node-id", node_id])
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    model.main(argv)
                report = json.loads((output / "report.json").read_text())
                self.assertAlmostEqual(report["predictions"][0]["expected_sync_us"],
                                       expected)
                self.assertEqual(report["sample_count"], count)
                self.assertEqual(report["node_id_filter"], node_id)
                self.assertEqual(report["validation"][0]["node_count"], 2)
                self.assertEqual(report["validation"][0]["observed_sync_us"], 20)
                if node_id:
                    with (output / "cdf.csv").open() as stream:
                        self.assertEqual([float(r["t_us"]) for r in csv.DictReader(stream)], [0, 2, 4])
                    with (output / "node_summary.csv").open() as stream:
                        self.assertEqual([r["node_id"] for r in csv.DictReader(stream)], ["01"])
            output = root / "missing"
            with contextlib.redirect_stderr(io.StringIO()) as stderr, self.assertRaises(SystemExit) as error:
                model.main(["--input", str(train), "--node-id", "missing", "--nodes", "2",
                            "--output-dir", str(output)])
            self.assertEqual(error.exception.code, 2)
            self.assertIn("no training samples match", stderr.getvalue())
            self.assertFalse(output.exists())

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
