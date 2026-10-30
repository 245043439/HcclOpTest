"""Compare per-node launch-delay CDFs and PDFs using Seaborn KDE plots."""

import argparse
from collections import defaultdict
from pathlib import Path

from node_sync_model import read_samples


def plot_distributions(samples, output_dir):
    """Overlay nodes on separate A/B panels, with consistent colors across plots."""
    import seaborn as sns
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    groups = defaultdict(list)
    for _, node, a, b in samples:
        groups[node].append((a, b))
    if not groups:
        raise ValueError("no samples to plot")
    nodes = sorted(groups)
    colors = dict(zip(nodes, sns.color_palette("husl", len(nodes))))
    with sns.axes_style("whitegrid"), sns.plotting_context("notebook"):
        for cumulative, filename, title, ylabel in [
                (True, "cdf.png", "CDF (cumulative KDE)", "Cumulative probability"),
                (False, "pdf.png", "PDF (KDE)", "Probability density (1/us)")]:
            figure = Figure(figsize=(13, 5), layout="constrained")
            FigureCanvasAgg(figure)
            axes = figure.subplots(1, 2)
            for column, (axis, name) in enumerate(zip(axes, ["A: earliest launch", "B: latest launch"])):
                all_constant = True
                values_all = []
                for node in nodes:
                    values = [pair[column] for pair in groups[node]]
                    values_all.extend(values)
                    label = f"node {node} (n={len(values)})"
                    if min(values) == max(values):
                        if cumulative:
                            sns.ecdfplot(x=values, ax=axis, color=colors[node],
                                         label=f"{label}; point mass")
                        else:
                            axis.axvline(values[0], color=colors[node], linestyle="--",
                                         label=f"{label}; point mass (no finite PDF)")
                    else:
                        all_constant = False
                        sns.kdeplot(x=values, cumulative=cumulative, bw_method="scott",
                                    bw_adjust=1, cut=4, gridsize=512, ax=axis,
                                    color=colors[node], label=label)
                axis.set(xlabel="Launch delay (us)", ylabel=ylabel, title=name)
                if cumulative:
                    axis.set_ylim(-0.03, 1.03)
                if all_constant:
                    lower, upper = min(values_all), max(values_all)
                    margin = (upper - lower) * 0.05 if upper > lower else max(abs(lower) * 0.05, 0.5)
                    axis.set_xlim(lower - margin, upper + margin)
                axis.legend(fontsize="small")
            figure.suptitle(f"{title} — comparison of {len(nodes)} nodes")
            figure.savefig(output_dir / filename, dpi=160)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="sample CSV: iteration,node_id,A_us,B_us")
    parser.add_argument("--node-ids", nargs="+", help="node IDs to compare; default: all nodes")
    parser.add_argument("--output-dir", required=True, type=Path, help="new directory for cdf.png and pdf.png")
    args = parser.parse_args(argv)
    try:
        samples = read_samples(args.input)
        if args.node_ids is not None:
            selected = set(args.node_ids)
            missing = selected - {sample[1] for sample in samples}
            if missing:
                raise ValueError(f"node IDs not found: {', '.join(sorted(missing))}")
            samples = [sample for sample in samples if sample[1] in selected]
        try:
            import seaborn
            import scipy
        except ImportError as exc:
            raise ValueError("plotting requires seaborn, matplotlib and scipy: "
                             "pip install -r node_sync_model/requirements.txt") from exc
        args.output_dir.mkdir(parents=True, exist_ok=False)
        plot_distributions(samples, args.output_dir)
    except (OSError, ValueError, OverflowError) as exc:
        parser.error(str(exc))
    print(f"Plots: {args.output_dir / 'cdf.png'}, {args.output_dir / 'pdf.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
