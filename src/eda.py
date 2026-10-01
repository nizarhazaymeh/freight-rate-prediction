"""Exploratory analysis: prints data-quality findings and saves figures.

Usage: python -m src.eda
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .data import load_csv

INK, ACCENT, MUTED = "#064A56", "#D9822B", "#9DAFB3"


def _style(ax, title):
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E3E8EA", linewidth=0.8)


def main() -> None:
    t = load_csv(config.TRAIN_PATH)
    v = load_csv(config.VALIDATION_PATH)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    t["rpm"] = t[config.TARGET] / t["distance"]
    lane_rpm = t.groupby(["pickup", "delivery", "equipment"])["rpm"].transform("median")
    ratio = t["rpm"] / lane_rpm
    corrupted = (ratio > config.OUTLIER_FACTOR) | (ratio < 1 / config.OUTLIER_FACTOR)

    train_cities = set(t["pickup"]) | set(t["delivery"])
    new_cities = sorted((set(v["pickup"]) | set(v["delivery"])) - train_cities)
    on_new = v["pickup"].isin(new_cities) | v["delivery"].isin(new_cities)
    seen_lanes = set(zip(t["pickup"], t["delivery"]))

    print("Data-quality findings")
    print(f"  train rows {len(t):,} ({t.date.min():%Y-%m-%d}..{t.date.max():%Y-%m-%d}), "
          f"validation rows {len(v):,} ({v.date.min():%Y-%m-%d}..{v.date.max():%Y-%m-%d})")
    print(f"  negative weights: train {(t.weight < 0).sum()}, validation {(v.weight < 0).sum()}")
    print(f"  missing weight: train {t.weight.isna().sum()}, validation {v.weight.isna().sum()}")
    print(f"  missing market_index: train {t.market_index.isna().sum()}, validation {v.market_index.isna().sum()}")
    print(f"  corrupted labels (>{config.OUTLIER_FACTOR}x off lane median): {corrupted.sum()} ({100 * corrupted.mean():.2f}%)")
    print(f"  validation cities unseen in training: {len(new_cities)} {new_cities} -> {on_new.sum():,} rows")
    print(f"  validation rows on a lane seen in training: "
          f"{100 * np.mean([l in seen_lanes for l in zip(v.pickup, v.delivery)]):.1f}%")
    mi = t.groupby("date")["market_index"]
    print(f"  market_index std within a day {mi.std().mean():.3f} vs between days {mi.mean().std():.3f}")

    # 1. Rate vs distance, corrupted labels highlighted.
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    ax.scatter(t.distance[~corrupted], t[config.TARGET][~corrupted], s=2, alpha=0.15, color=INK, label="clean")
    ax.scatter(t.distance[corrupted], t[config.TARGET][corrupted], s=6, alpha=0.8, color=ACCENT,
               label=f"corrupted label ({corrupted.sum()})")
    ax.set_xlabel("Distance (miles)"); ax.set_ylabel("Posted rate ($)"); ax.legend(frameon=False)
    _style(ax, "Rate scales with distance; 1.4% of labels are 0.2-0.5x or 2.5-4.6x off")
    fig.tight_layout(); fig.savefig(config.FIGURES_DIR / "rate_vs_distance.png"); plt.close(fig)

    # 2. Ratio to lane median (log scale) - shows the clean separation.
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=150)
    ax.hist(np.log10(ratio.clip(0.1, 10)), bins=120, color=INK)
    ax.set_yscale("log"); ax.set_xlabel("log10(rate / lane-equipment median)"); ax.set_ylabel("rows (log)")
    for x in (np.log10(config.OUTLIER_FACTOR), -np.log10(config.OUTLIER_FACTOR)):
        ax.axvline(x, color=ACCENT, linestyle="--", linewidth=1)
    _style(ax, "Corrupted labels form separate clusters far from genuine noise")
    fig.tight_layout(); fig.savefig(config.FIGURES_DIR / "label_ratio_hist.png"); plt.close(fig)

    # 3. Daily market index and daily rate per mile.
    clean = t[~corrupted]
    daily = clean.groupby("date").agg(rpm=("rpm", "median"), mi=("market_index", "mean"))
    all_mi = pd.concat([t, v]).groupby("date")["market_index"].mean()
    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=150)
    ax.plot(daily.index, daily.rpm.rolling(7, center=True).mean(), color=INK, label="median $/mile (7-day)")
    ax2 = ax.twinx()
    ax2.plot(all_mi.index, all_mi.rolling(7, center=True).mean(), color=ACCENT, label="market_index (7-day)")
    ax2.axvspan(pd.Timestamp("2025-11-01"), all_mi.index.max(), color=MUTED, alpha=0.15)
    ax2.text(pd.Timestamp("2025-11-05"), ax2.get_ylim()[1] * 0.97, "validation", color="#455A60", fontsize=8, va="top")
    ax.set_ylabel("$/mile"); ax2.set_ylabel("market_index")
    ax.spines[["top"]].set_visible(False); ax2.spines[["top"]].set_visible(False)
    ax.set_title("market_index is a daily market factor driving rate per mile", loc="left", fontsize=12, fontweight="bold")
    fig.legend(loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(config.FIGURES_DIR / "market_index_vs_rpm.png"); plt.close(fig)

    # 4. Equipment premium and weight effect.
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), dpi=150)
    clean.groupby("equipment")["rpm"].median().reindex(config.EQUIPMENT).plot.bar(ax=axes[0], color=INK, rot=0)
    axes[0].set_ylabel("median $/mile"); axes[0].set_xlabel("")
    _style(axes[0], "Equipment premium")
    rel = (clean["rpm"] / lane_rpm[~corrupted])
    wb = clean.groupby(pd.qcut(clean.weight.abs(), 10), observed=True).apply(lambda g: rel.loc[g.index].mean())
    axes[1].plot(range(len(wb)), wb.values, marker="o", color=INK)
    axes[1].set_xticks(range(len(wb))); axes[1].set_xticklabels([f"{i.right / 1000:.0f}k" for i in wb.index], fontsize=7)
    axes[1].set_xlabel("weight decile (upper bound, lb)"); axes[1].set_ylabel("rate / lane median")
    _style(axes[1], "Heavier loads cost more")
    fig.tight_layout(); fig.savefig(config.FIGURES_DIR / "equipment_weight.png"); plt.close(fig)

    print(f"Saved figures to {config.FIGURES_DIR.relative_to(config.ROOT)}/")


if __name__ == "__main__":
    main()
