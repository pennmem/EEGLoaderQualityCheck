"""Plotting utilities for validation results."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def plot_comparison_results(
    df_results: pd.DataFrame,
    col_tgt: str,
    col_std: Optional[str] = None,
    col_label: Optional[str] = None,
    title: Optional[str] =None
):
    """Plot a metric across sessions per subject, grouped by comparison × experiment."""
    comparisons = df_results["comparison"].unique()
    subjects = df_results["subject"].unique()
    experiments = df_results["experiment"].unique()

    for experiment in experiments:
        n = len(comparisons)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharex=True)
        if n == 1:
            axes = [axes]

        for i, comp in enumerate(comparisons):
            ax = axes[i]
            mask = (df_results["comparison"] == comp) & (df_results["experiment"] == experiment)
            comp_df = df_results[mask]

            for subj in subjects:
                subj_df = comp_df[comp_df["subject"] == subj].sort_values("session")
                if subj_df.empty:
                    continue

                (line,) = ax.plot(subj_df["session"], subj_df[col_tgt], marker="o", label=subj)

                if col_std is not None and col_std in subj_df.columns:
                    ax.fill_between(
                        subj_df["session"],
                        subj_df[col_tgt] - subj_df[col_std],
                        subj_df[col_tgt] + subj_df[col_std],
                        color=line.get_color(),
                        alpha=0.15,
                    )
            plot_title = f"{title}: {experiment} | {comp}" if title is not None else f"{experiment} | {comp}"
            ax.set_title(plot_title)
            ax.set_xlabel("Session")
            ylabel = col_label or col_tgt
            if col_std:
                ylabel += r" ($\pm$ Std)"
            if i == 0:
                ax.set_ylabel(ylabel)
            ax.legend(title="Subject")

        plt.tight_layout()
        plt.show()
