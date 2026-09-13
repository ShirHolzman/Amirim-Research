"""
Figures for 02_reliability, drawn ONLY from reliability.csv and reliability_summary.csv --
no cache/, no catie/ import, no recomputation. A bug here can produce an ugly picture;
it can never move a number (see sub_plans/02_reliability.md).

Five figures:
    fig1_baseline_biased.png     P(biased alt) vs outcome, training -- superseded, kept
                                  as the before/after reference for the supervisor's ask
    fig2_doubled_main.png        the label-equivariant diagram, 10 bins, training
    fig3_doubled_bin_sweep.png   the same set at 10/20/50/100 (uniform) and 10/20 (quantile)
    fig4_doubled_run_length.png  the doubled diagram stratified by run length -- does the
                                  gap reverse sign as the streak gets longer
    fig5_doubled_eda.png         one replication panel on EDA

Run:  cd my_code/catie_calibration && python 02_reliability/figures.py
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")  # before pyplot, per repo convention
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = pathlib.Path(__file__).parent
OUT_DIR = HERE / "figures"
OUT_DIR.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "font.size": 10})


def _panel(ax, table: pd.DataFrame, title: str, summary_row=None):
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1, zorder=1)
    ax.errorbar(table["predicted"], table["empirical"],
               yerr=[table["empirical"] - table["emp_lo"], table["emp_hi"] - table["empirical"]],
               fmt="o", color="#1f77b4", ecolor="#1f77b4", elinewidth=1, capsize=2,
               markersize=4, zorder=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("predicted")
    ax.set_ylabel("empirical")
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    if summary_row is not None:
        title += (f"\nn={int(summary_row['n']):,}  ECE={summary_row['ece']:.3f}"
                 f"  E[p]={summary_row['e_p']:.3f}  E[log p]={summary_row['e_log_p']:.3f}")
    ax.set_title(title, fontsize=9)


def _curve(bins, summary, diagram, split, stratum, n_bins, strategy):
    t = bins[(bins.diagram == diagram) & (bins.split == split) & (bins.stratum == stratum) &
             (bins.n_bins == n_bins) & (bins.strategy == strategy)].sort_values("bin")
    s = summary[(summary.diagram == diagram) & (summary.split == split) &
                (summary.stratum == stratum) & (summary.n_bins == n_bins) &
                (summary.strategy == strategy)]
    return t, (s.iloc[0] if len(s) else None)


def fig1_baseline(bins, summary):
    t, s = _curve(bins, summary, "baseline_biased", "training", "all", 10, "uniform")
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    _panel(ax, t, "P(biased alt) vs outcome -- training\n(SUPERSEDED, see fig2)", s)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig1_baseline_biased.png", bbox_inches="tight")
    plt.close(fig)


def fig2_doubled_main(bins, summary):
    t, s = _curve(bins, summary, "doubled", "training", "all", 10, "uniform")
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    _panel(ax, t, "Label-equivariant diagram -- training, 10 bins", s)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_doubled_main.png", bbox_inches="tight")
    plt.close(fig)


def fig3_bin_sweep(bins, summary):
    members = [(10, "uniform"), (20, "uniform"), (50, "uniform"),
              (100, "uniform"), (10, "quantile"), (20, "quantile")]
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.4))
    for ax, (n_bins, strategy) in zip(axes.ravel(), members):
        t, s = _curve(bins, summary, "doubled", "training", "all", n_bins, strategy)
        _panel(ax, t, f"{strategy}, {n_bins} bins", s)
    fig.suptitle("Doubled diagram -- bin-count / strategy sweep, training", y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig3_doubled_bin_sweep.png", bbox_inches="tight")
    plt.close(fig)


def fig4_run_length(bins, summary):
    # No summary_row passed to _panel: five narrow panels have no room for the
    # n/ECE/E[p]/E[log p] line without titles colliding -- those numbers are in
    # reliability_summary.csv and quoted in README.md instead.
    strata = ["1", "2", "3-4", "5-9", "10+"]
    fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=True)
    for ax, stratum in zip(axes, strata):
        t, _ = _curve(bins, summary, f"doubled_run{stratum}", "training", stratum, 10, "uniform")
        _panel(ax, t, f"run length {stratum}")
    fig.suptitle("Doubled diagram by run length -- history-only, training", y=1.05)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig4_doubled_run_length.png", bbox_inches="tight")
    plt.close(fig)


def fig5_eda(bins, summary):
    t, s = _curve(bins, summary, "doubled_eda", "eda", "all", 10, "uniform")
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    _panel(ax, t, "Label-equivariant diagram -- EDA replication\n(generated last, no decision follows it)", s)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig5_doubled_eda.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    bins = pd.read_csv(HERE / "reliability.csv")
    summary = pd.read_csv(HERE / "reliability_summary.csv")
    fig1_baseline(bins, summary)
    fig2_doubled_main(bins, summary)
    fig3_bin_sweep(bins, summary)
    fig4_run_length(bins, summary)
    fig5_eda(bins, summary)


if __name__ == "__main__":
    main()
