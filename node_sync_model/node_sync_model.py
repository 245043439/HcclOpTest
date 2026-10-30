"""Estimate node launch CDFs and integrate the expected global launch span (us)."""

import argparse
from bisect import bisect_right
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import sys


def read_samples(path):
    """Read one paired observation per (iteration, node_id); allow extra columns."""
    samples = []
    seen = set()
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"iteration", "node_id", "A_us", "B_us"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: required columns: {', '.join(sorted(required))}")
        for row in reader:
            try:
                iteration = row["iteration"].strip()
                node = row["node_id"].strip()
                a, b = float(row["A_us"]), float(row["B_us"])
                if not iteration or not node:
                    raise ValueError("empty iteration or node_id")
                if not math.isfinite(a) or not math.isfinite(b) or a > b:
                    raise ValueError("A_us and B_us must be finite and A_us <= B_us")
                if (iteration, node) in seen:
                    raise ValueError("duplicate (iteration, node_id)")
            except (AttributeError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{reader.line_num}: {exc}") from exc
            seen.add((iteration, node))
            samples.append((iteration, node, a, b))
    if not samples:
        raise ValueError(f"{path}: no samples")
    return samples


def empirical_cdf(samples):
    """Return right-continuous marginal CDFs on their shared sorted support."""
    a = sorted(sample[2] for sample in samples)
    b = sorted(sample[3] for sample in samples)
    return [(t, bisect_right(a, t) / len(a), bisect_right(b, t) / len(b))
            for t in sorted(set(a + b))]


def expected_span(cdf, nodes):
    """Exact step-function integral; expm1 avoids cancellation near zero."""
    if isinstance(nodes, bool) or not isinstance(nodes, int) or nodes < 1:
        raise ValueError("node count must be a positive integer")
    contributions = []
    for (t, fa, fb), (next_t, _, _) in zip(cdf, cdf[1:]):
        # Evaluate 1 - the larger power first, keeping small spans accurate.
        if fb == 0.0 and fa == 1.0:
            coverage = 1.0
        elif fb >= 1.0 - fa:
            coverage = -math.expm1(nodes * math.log(fb)) - (1.0 - fa) ** nodes
        else:
            coverage = -math.expm1(nodes * math.log1p(-fa)) - fb ** nodes
        contributions.append((next_t - t) * max(0.0, min(1.0, coverage)))
    result = math.fsum(contributions)
    if not math.isfinite(result):
        raise ValueError("sample time range exceeds finite floating-point arithmetic")
    return result


def observed_span(samples):
    """Require the same node roster every round for scale validation."""
    rounds = defaultdict(list)
    for iteration, node, a, b in samples:
        rounds[iteration].append((node, a, b))
    roster = {row[0] for row in next(iter(rounds.values()))}
    if any({row[0] for row in rows} != roster for rows in rounds.values()):
        raise ValueError("validation requires the same node IDs in every iteration")
    spans = [max(row[2] for row in rows) - min(row[1] for row in rows)
             for rows in rounds.values()]
    return len(roster), len(rounds), math.fsum(spans) / len(spans)


def node_summary(samples):
    groups = defaultdict(list)
    for _, node, a, b in samples:
        groups[node].append((a, b))
    return [{"node_id": node, "sample_count": len(pairs),
             "mean_A_us": math.fsum(a / len(pairs) for a, _ in pairs),
             "mean_B_us": math.fsum(b / len(pairs) for _, b in pairs),
             "min_A_us": min(a for a, _ in pairs),
             "max_B_us": max(b for _, b in pairs)}
            for node, pairs in sorted(groups.items())]


def prediction_with_error(samples, nodes, predicted):
    """Compare to the mean per-step span of IDs 0..N-1 in unfiltered input."""
    rounds = defaultdict(dict)
    for iteration, node, a, b in samples:
        rounds[iteration][node] = (a, b)
    result = {"node_count": nodes, "expected_sync_us": predicted,
              "y_true_mean_us": None, "error_us": None,
              "absolute_error_us": None, "relative_error": None,
              "iteration_count": len(rounds), "error_status": "missing_required_nodes"}
    if any(len(rows) < nodes for rows in rounds.values()):
        return result
    required = [str(i) for i in range(nodes)]
    if any(any(node not in rows for node in required) for rows in rounds.values()):
        return result
    spans = [max(rows[node][1] for node in required)
             - min(rows[node][0] for node in required) for rows in rounds.values()]
    observed = math.fsum(span / len(spans) for span in spans)
    error = predicted - observed
    result.update(y_true_mean_us=observed, error_us=error, absolute_error_us=abs(error),
                  relative_error=error / observed if observed else None, error_status="ok")
    return result


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="training sample CSV")
    parser.add_argument("--node-id", help="fit CDF using only this training node ID (exact string match)")
    parser.add_argument("--nodes", required=True, nargs="+", type=int,
                        help="positive target node counts, e.g. 1 2 4 8")
    parser.add_argument("--validation", action="append", default=[], type=Path,
                        help="held-out sample CSV at one scale; repeat for more scales")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="new directory for CDF, predictions and validation")
    args = parser.parse_args(argv)
    try:
        if any(n < 1 for n in args.nodes):
            raise ValueError("--nodes values must be positive")
        all_samples = read_samples(args.input)
        samples = all_samples
        if args.node_id is not None:
            samples = [sample for sample in samples if sample[1] == args.node_id]
            if not samples:
                raise ValueError(f"no training samples match --node-id {args.node_id!r}")
        cdf = empirical_cdf(samples)
        predictions = [prediction_with_error(all_samples, n, expected_span(cdf, n))
                       for n in sorted(set(args.nodes))]
        validation = []
        for path in args.validation:
            if path.resolve() == args.input.resolve():
                raise ValueError("validation must use a separate held-out sample file")
            n, rounds, observed = observed_span(read_samples(path))
            predicted = expected_span(cdf, n)
            validation.append({"source": str(path), "node_count": n,
                               "iteration_count": rounds, "observed_sync_us": observed,
                               "predicted_sync_us": predicted,
                               "error_us": predicted - observed,
                               "relative_error": ((predicted - observed) / observed
                                                  if observed else None)})
        report = {
            "input": str(args.input), "sample_count": len(samples),
            "node_id_filter": args.node_id,
            "iteration_count": len({row[0] for row in samples}),
            "training_node_count": len({row[1] for row in samples}),
            "empirical_span_limit_us": cdf[-1][0] - cdf[0][0],
            "predictions": predictions, "validation": validation,
            "assumptions": ["Common launch reference across nodes",
                            "IID node pairs; stable distribution across target scales",
                            "Input contains one operator/size/mode/ranks-per-node group"],
            "limitations": ["Empirical CDF cannot extrapolate beyond observed tails",
                            "Node summaries do not establish IID or temporal independence",
                            "No confidence intervals or P95/P99 are estimated"],
        }
        encoded_report = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        write_csv(args.output_dir / "cdf.csv", ["t_us", "F_A", "F_B"],
                  [dict(zip(["t_us", "F_A", "F_B"], row)) for row in cdf])
        write_csv(args.output_dir / "predictions.csv",
                  list(predictions[0]), predictions)
        summaries = node_summary(samples)
        write_csv(args.output_dir / "node_summary.csv", list(summaries[0]), summaries)
        if validation:
            write_csv(args.output_dir / "validation.csv", list(validation[0]), validation)
        (args.output_dir / "report.json").write_text(encoded_report + "\n", encoding="utf-8")
    except (OSError, ValueError, OverflowError) as exc:
        parser.error(str(exc))
    writer = csv.DictWriter(sys.stdout, fieldnames=list(predictions[0]))
    writer.writeheader()
    writer.writerows(predictions)
    print(f"Results: {args.output_dir}", file=sys.stderr)
    print("Empirical predictions are bounded by the observed sample range; "
          "check tail coverage before extrapolation.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
