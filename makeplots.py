import csv
import math
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wrlattices import wr_field_indicator, pair_count_sieve

FIGDIR = os.path.join("..", "paper", "figures")
DELTA = 1.0 - (1.0 + math.log(math.log(2.0))) / math.log(2.0)

COLORS = {"sieve": "#2a78d6", "spf": "#1baf7a", "divscan": "#eda100",
          "classify": "#008300", "brute": "#4a3aa7"}
MARKERS = {"sieve": "o", "spf": "s", "divscan": "^", "classify": "D", "brute": "v"}
STYLES = {"sieve": "-", "spf": "--", "divscan": "-.", "classify": "-", "brute": ":"}
LABELS = {"sieve": "pair sieve", "spf": "factor-table scan", "divscan": "divisor scan",
          "classify": "classification method", "brute": "ideal search"}

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.4,
    "lines.linewidth": 1.3,
    "text.color": "#0b0b0b",
    "axes.edgecolor": "#52514e",
    "axes.labelcolor": "#0b0b0b",
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
})


def read_rows(name):
    with open(name) as f:
        return list(csv.DictReader(f))


def density_figure():
    rows = read_rows("results/density.csv")
    N = np.array([float(r["N"]) for r in rows])
    wr = np.array([float(r["wr_fields"]) for r in rows])
    sfodd = np.array([float(r["sf_odd"]) for r in rows])
    sfall = 1.5 * sfodd
    field_ratio = wr / sfall
    pair_ratio = np.array([float(r["pair_ratio"]) for r in rows])
    shape = np.array([float(r["ford_shape"]) for r in rows])
    cfit = field_ratio[-1] / shape[-1]
    claimed = (math.sqrt(3.0) - 1.0) / (2.0 * math.sqrt(3.0))
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    ax.set_xscale("log")
    ax.plot(N, pair_ratio, "-.", color="#eda100", marker="^", markersize=3.6,
            markerfacecolor="white", markeredgewidth=0.8, markeredgecolor="#eda100",
            label=r"divisor pairs per integer")
    ax.axhline(claimed, color="#e34948", linestyle=":", linewidth=1.0,
               label=r"claimed lower bound $(\sqrt{3}-1)/(2\sqrt{3})$")
    ax.plot(N, field_ratio, "-", color="#2a78d6", marker="o", markersize=3.6,
            markerfacecolor="white", markeredgewidth=0.8, markeredgecolor="#2a78d6",
            label="proportion of fields containing WR ideals")
    ax.plot(N, cfit * shape, "--", color="#52514e", linewidth=0.9,
            label=r"$c\,(\log N)^{-\delta}(\log\log N)^{-3/2}$")
    ax.set_xlabel("$N$")
    ax.set_ylabel("ratio")
    ax.set_ylim(0.0, 0.62)
    ax.grid(True, which="major", color="#e6e5e0")
    ax.legend(frameon=False, loc="center left", handlelength=2.2)
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(FIGDIR, "density.eps"), format="eps")
    plt.close(fig)


def range_figure():
    rows = [r for r in read_rows("results/benchmark.csv") if r["task"] == "range"]
    fig, ax = plt.subplots(figsize=(5.4, 3.1))
    ax.set_xscale("log")
    ax.set_yscale("log")
    for method in ("divscan", "spf", "sieve"):
        pts = sorted((int(r["size"]), float(r["seconds"])) for r in rows if r["method"] == method)
        ax.plot([p[0] for p in pts], [max(p[1], 1e-4) for p in pts], STYLES[method],
                color=COLORS[method], marker=MARKERS[method], markersize=3.6,
                markerfacecolor="white", markeredgewidth=0.8,
                markeredgecolor=COLORS[method], label=LABELS[method])
    ax.set_xlabel("$N$")
    ax.set_ylabel("wall time (s)")
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 80)
    ax.grid(True, which="major", color="#e6e5e0")
    ax.legend(frameon=False, loc="upper left", handlelength=2.2)
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(FIGDIR, "bench_range.eps"), format="eps")
    plt.close(fig)


def field_figure():
    rows = [r for r in read_rows("results/benchmark.csv") if r["task"] == "field"]
    fig, ax = plt.subplots(figsize=(5.4, 3.1))
    ax.set_xscale("log")
    ax.set_yscale("log")
    for method in ("brute", "classify"):
        pts = sorted((int(r["size"]), float(r["seconds"])) for r in rows if r["method"] == method)
        ax.plot([p[0] for p in pts], [max(p[1], 1e-5) for p in pts], STYLES[method],
                color=COLORS[method], marker=MARKERS[method], markersize=3.6,
                markerfacecolor="white", markeredgewidth=0.8,
                markeredgecolor=COLORS[method], label=LABELS[method])
    ax.set_xlabel("$D$")
    ax.set_ylabel("wall time (s)")
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 80)
    ax.grid(True, which="major", color="#e6e5e0")
    ax.legend(frameon=False, loc="upper left", handlelength=2.2)
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(FIGDIR, "bench_field.eps"), format="eps")
    plt.close(fig)


def rho_figure():
    N = 10 ** 7
    ind = wr_field_indicator(N)
    pc = pair_count_sieve(N)
    vals = pc[np.nonzero(ind)[0]]
    top = int(vals.max())
    counts = np.bincount(vals, minlength=top + 1)[1:top + 1]
    fig, ax = plt.subplots(figsize=(5.0, 2.9))
    ax.bar(np.arange(1, top + 1), counts, color="#2a78d6", edgecolor="white", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(1, top + 1))
    ax.set_xlabel(r"$\rho_{3}(D)$")
    ax.set_ylabel("number of fields")
    ax.grid(True, which="major", axis="y", color="#e6e5e0")
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(FIGDIR, "rho_hist.eps"), format="eps")
    plt.close(fig)
    return counts


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    density_figure()
    range_figure()
    field_figure()
    counts = rho_figure()
    print("rho distribution at 1e7:", list(counts))
    print("figures written to", FIGDIR)


if __name__ == "__main__":
    main()
