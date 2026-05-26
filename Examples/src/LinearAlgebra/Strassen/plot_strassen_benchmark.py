#!/usr/bin/env python3
"""
Plot the CSV output produced by main_StrassenBenchmark.

The generated figure contains the full measured dataset:
- Eigen median times for every tested size
- Strassen median times for every size/cutoff pair
- speedup heatmap for every size/cutoff pair
- relative error heatmap for every size/cutoff pair
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class BenchmarkRow:
    size: int
    method: str
    cutoff: int | None
    min_ms: float
    median_ms: float
    max_ms: float
    speedup: float
    rel_error: float


def parse_benchmark(path: Path) -> list[BenchmarkRow]:
    rows: list[BenchmarkRow] = []

    with path.open("r", encoding="utf-8", newline="") as stream:
        filtered_lines = (
            line for line in stream if line.strip() and not line.lstrip().startswith("#")
        )
        reader = csv.DictReader(filtered_lines)
        for record in reader:
            cutoff = None if record["cutoff"] == "NA" else int(record["cutoff"])
            rows.append(
                BenchmarkRow(
                    size=int(record["size"]),
                    method=record["method"],
                    cutoff=cutoff,
                    min_ms=float(record["min_ms"]),
                    median_ms=float(record["median_ms"]),
                    max_ms=float(record["max_ms"]),
                    speedup=float(record["speedup"]),
                    rel_error=float(record["rel_error"]),
                )
            )

    if not rows:
        raise ValueError("No benchmark rows found in the input file")

    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot the full Strassen benchmark dataset."
    )
    parser.add_argument("input", type=Path, help="benchmark CSV output file")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("strassen_benchmark.png"),
        help="output image path",
    )
    return parser


def build_heatmap_data(
    rows: list[BenchmarkRow],
) -> tuple[list[int], list[int], np.ndarray, np.ndarray, np.ndarray]:
    sizes = sorted({row.size for row in rows})
    cutoffs = sorted({row.cutoff for row in rows if row.cutoff is not None})

    time_map = np.full((len(cutoffs), len(sizes)), np.nan)
    speedup_map = np.full((len(cutoffs), len(sizes)), np.nan)
    error_map = np.full((len(cutoffs), len(sizes)), np.nan)

    size_to_col = {size: idx for idx, size in enumerate(sizes)}
    cutoff_to_row = {cutoff: idx for idx, cutoff in enumerate(cutoffs)}

    for row in rows:
        if row.method != "Strassen" or row.cutoff is None:
            continue
        r = cutoff_to_row[row.cutoff]
        c = size_to_col[row.size]
        time_map[r, c] = row.median_ms
        speedup_map[r, c] = row.speedup
        error_map[r, c] = row.rel_error

    return sizes, cutoffs, time_map, speedup_map, error_map


def plot_heatmap(
    axis: plt.Axes,
    data: np.ndarray,
    sizes: list[int],
    cutoffs: list[int],
    title: str,
    colorbar_label: str,
    cmap: str,
    log_scale: bool = False,
) -> None:
    valid = data[np.isfinite(data)]
    if valid.size == 0:
        raise ValueError(f"No valid data available for {title}")

    plot_data = data.copy()
    if log_scale:
        positive = plot_data[np.isfinite(plot_data) & (plot_data > 0.0)]
        floor = positive.min() if positive.size else np.finfo(float).tiny
        plot_data[~np.isfinite(plot_data)] = np.nan
        plot_data[(plot_data <= 0.0) & np.isfinite(plot_data)] = floor
        plot_data = np.log10(plot_data)
        colorbar_label = f"log10({colorbar_label})"

    image = axis.imshow(plot_data, aspect="auto", origin="lower", cmap=cmap)
    axis.set_title(title)
    axis.set_xlabel("Matrix size")
    axis.set_ylabel("Cutoff")
    axis.set_xticks(range(len(sizes)), labels=sizes, rotation=45, ha="right")
    axis.set_yticks(range(len(cutoffs)), labels=cutoffs)

    for row in range(len(cutoffs)):
        for col in range(len(sizes)):
            value = data[row, col]
            if np.isfinite(value):
                label = f"{value:.2e}" if log_scale else f"{value:.2f}"
                axis.text(col, row, label, ha="center", va="center", fontsize=8)

    plt.colorbar(image, ax=axis, label=colorbar_label)


def plot_results(rows: list[BenchmarkRow], output: Path) -> None:
    eigen_rows = sorted(
        (row for row in rows if row.method == "Eigen"), key=lambda row: row.size
    )
    if not eigen_rows:
        raise ValueError("No Eigen rows found in the input file")

    sizes, cutoffs, strassen_time, speedup_map, error_map = build_heatmap_data(rows)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    ax_eigen, ax_time = axes[0]
    ax_speedup, ax_error = axes[1]

    ax_eigen.plot(
        [row.size for row in eigen_rows],
        [row.median_ms for row in eigen_rows],
        marker="o",
        linewidth=2.0,
        color="#1f77b4",
    )
    ax_eigen.fill_between(
        [row.size for row in eigen_rows],
        [row.min_ms for row in eigen_rows],
        [row.max_ms for row in eigen_rows],
        alpha=0.2,
        color="#1f77b4",
    )
    ax_eigen.set_title("Eigen Median Time")
    ax_eigen.set_xlabel("Matrix size")
    ax_eigen.set_ylabel("Time [ms]")
    ax_eigen.set_yscale("log")
    ax_eigen.grid(True, which="both", linestyle="--", alpha=0.35)

    plot_heatmap(
        ax_time,
        strassen_time,
        sizes,
        cutoffs,
        title="Strassen Median Time",
        colorbar_label="time [ms]",
        cmap="viridis",
        log_scale=True,
    )
    plot_heatmap(
        ax_speedup,
        speedup_map,
        sizes,
        cutoffs,
        title="Speedup (Eigen / Strassen)",
        colorbar_label="speedup",
        cmap="coolwarm",
    )
    plot_heatmap(
        ax_error,
        error_map,
        sizes,
        cutoffs,
        title="Relative Error",
        colorbar_label="relative error",
        cmap="magma",
        log_scale=True,
    )

    fig.suptitle("Strassen Benchmark: Complete Dataset")
    fig.savefig(output, dpi=160)


def main() -> int:
    args = build_parser().parse_args()
    rows = parse_benchmark(args.input)
    plot_results(rows, args.output)
    print(f"Wrote plot to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
