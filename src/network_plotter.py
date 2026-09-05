"""Render the curtailment result matrix as a heatmap.

The study runner solves and publishes; this module is the only place that
knows how the published matrix is drawn, so the two can change independently.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

# Chosen before pyplot is imported, so the module renders without a display.
matplotlib.use("Agg")

import numpy as np

# ---------------------------------------------------------------------------
# The published tables
# ---------------------------------------------------------------------------


class NetworkPlotter:
    """The published outputs of a curtailment study: the table and the figure.

    Holds one result matrix, so the caller reads it once and renders from it.
    """

    def __init__(self, document: dict):
        self.document = document
        meta = document.get("meta") or {}
        # What to draw comes from the matrix, so a study of any feeders and
        # any control modes renders without this module being told about them.
        self.control_modes = list(meta.get("control_modes") or [])
        if not self.control_modes:
            self.control_modes = sorted(
                {
                    mode
                    for by_case in (document.get("results") or {}).values()
                    for by_mode in by_case.values()
                    for mode in by_mode
                }
            )
        self.case_labels = dict(meta.get("case_labels") or {})

    def _mode_label(self, mode: str) -> str:
        """Return the two-line column heading for one control mode."""
        for p_mode in ("CONSTANT_P", "MPPT"):
            if mode.startswith(p_mode + "_"):
                return f"{p_mode}\n{mode[len(p_mode) + 1 :]}"
        return mode

    @classmethod
    def from_file(cls, path: Path) -> NetworkPlotter:
        """Read a published result matrix from disk."""
        return cls(json.loads(Path(path).read_text()))

    def write_runtime_table(self, output_path: Path) -> None:
        """Write the per-case runtime and convergence table as CSV.

        One row per case, carrying what a reader needs to judge or repeat a single
        number: whether it solved, how the solve terminated, how many iterations it
        took, how long, and the residuals it finished on.
        """
        document = self.document
        columns = [
            "case",
            "p_mode",
            "q_mode",
            "norm",
            "solved",
            "termination",
            "iterations",
            "time_s",
            "warm_start",
            "linear_solver",
            "mumps_pivot_tolerance",
            "initial_delivery_fraction",
            "max_constraint_violation",
            "constraint_allowance_ratio",
            "max_tsbi_residual",
            "max_voltage_band_violation_pu",
            "max_line_ampacity_ratio",
            "max_inverter_apparent_power_ratio",
            "curtailed_w",
            "delivered_w",
        ]
        rows = []
        for norm, cases in (document.get("results") or {}).items():
            for case, modes in cases.items():
                for mode, entry in modes.items():
                    quality = entry.get("quality") or {}
                    settings = entry.get("solve_settings") or {}
                    p_mode, _, q_mode = mode.partition("_")
                    if p_mode == "CONSTANT":
                        p_mode, _, q_mode = (
                            "CONSTANT_P",
                            None,
                            mode[len("CONSTANT_P_") :],
                        )
                    rows.append(
                        {
                            "case": case,
                            "p_mode": p_mode,
                            "q_mode": q_mode,
                            "norm": norm,
                            "solved": entry.get("solved"),
                            "termination": (entry.get("msg") or "").splitlines()[0][
                                :60
                            ],
                            "iterations": entry.get("iterations"),
                            "time_s": entry.get("time_s"),
                            "warm_start": settings.get("warm_start"),
                            "linear_solver": settings.get("linear_solver"),
                            "mumps_pivot_tolerance": settings.get(
                                "mumps_pivot_tolerance"
                            ),
                            "initial_delivery_fraction": settings.get(
                                "initial_delivery_fraction"
                            ),
                            "max_constraint_violation": quality.get(
                                "max_model_constraint_violation"
                            ),
                            "constraint_allowance_ratio": quality.get(
                                "worst_constraint_allowance_ratio"
                            ),
                            "max_tsbi_residual": quality.get("max_tsbi_residual"),
                            "max_voltage_band_violation_pu": quality.get(
                                "max_voltage_band_violation_pu"
                            ),
                            "max_line_ampacity_ratio": quality.get(
                                "max_line_ampacity_ratio"
                            ),
                            "max_inverter_apparent_power_ratio": quality.get(
                                "max_inverter_apparent_power_ratio"
                            ),
                            "curtailed_w": entry.get("curtailed_w"),
                            "delivered_w": entry.get("delivered_w"),
                        }
                    )
        rows.sort(key=lambda r: (r["case"], r["p_mode"], r["q_mode"] or "", r["norm"]))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    def render_heatmap(self, output_path: Path, norm: str = "ALL") -> None:
        """Render the sorted curtailment heatmap of this result matrix."""
        import matplotlib.pyplot as plt

        document = self.document
        all_results = document.get("results", {})
        # The runner writes the norms in a stable order, so the file itself
        # carries the row order and no second list has to agree with it.
        selected_norms = tuple(all_results) if norm == "ALL" else (norm,)
        case_names = document.get("meta", {}).get("case_order") or list(
            all_results.get(selected_norms[0], {})
        )
        rows = [
            (case_name, row_norm)
            for case_name in case_names
            for row_norm in selected_norms
        ]
        values = np.full((len(rows), len(self.control_modes)), np.nan)
        status = [["missing"] * len(self.control_modes) for _ in rows]

        for row, (case_name, row_norm) in enumerate(rows):
            result_by_norm = all_results.get(row_norm, {})
            for col, mode in enumerate(self.control_modes):
                entry = result_by_norm.get(case_name, {}).get(mode)
                if not entry:
                    continue
                if entry.get("solved") and entry.get("requested_w", 0.0) > 0.0:
                    values[row, col] = (
                        100.0
                        * float(
                            entry.get("network_curtailment_w", entry["curtailed_w"])
                        )
                        / float(entry["requested_w"])
                    )
                    status[row][col] = "solved"
                else:
                    status[row][col] = "failed"

        cmap = plt.get_cmap("RdYlGn_r").copy()
        cmap.set_bad("#d9dee7")
        figure_height = 3.7 + 0.56 * len(rows)
        fig, ax = plt.subplots(figsize=(14.8, figure_height), dpi=180)
        fig.subplots_adjust(left=0.16, right=0.91, top=0.86, bottom=0.18)
        finite_values = values[np.isfinite(values)]
        if finite_values.size:
            observed_min = float(np.min(finite_values))
            observed_max = float(np.max(finite_values))
            if math.isclose(observed_min, observed_max, rel_tol=0.0, abs_tol=1e-12):
                # Avoid a singular color normalization when a filtered plot has
                # only one solved value.  For normal 120-cell plots the scale is
                # exactly the observed minimum-to-maximum range.
                padding = max(0.1, abs(observed_min) * 0.05)
                color_min = max(0.0, observed_min - padding)
                color_max = observed_max + padding
            else:
                color_min = observed_min
                color_max = observed_max
        else:
            observed_min = observed_max = 0.0
            color_min, color_max = 0.0, 1.0
        image = ax.imshow(
            values,
            cmap=cmap,
            vmin=color_min,
            vmax=color_max,
            aspect="auto",
        )

        ax.set_xticks(
            range(len(self.control_modes)),
            [self._mode_label(mode) for mode in self.control_modes],
            fontsize=9,
        )
        row_labels = [
            f"{self.case_labels.get(case_name, case_name)}  ·  "
            f"{row_norm.replace('LINF', 'L∞')}"
            for case_name, row_norm in rows
        ]
        ax.set_yticks(range(len(rows)), row_labels, fontsize=9.5)
        ax.tick_params(length=0, pad=8)
        ax.set_xlabel("Inverter active/reactive control mode", labelpad=12, fontsize=11)
        ax.set_ylabel("Feeder", labelpad=10, fontsize=11)
        ax.set_title(
            "Grid curtailment by feeder, control mode, and dispatch norm",
            fontsize=17,
            weight="bold",
            pad=20,
        )
        ax.text(
            0.5,
            1.015,
            "Percent of requested power curtailed; gray = solver failure",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=10,
            color="#4b5563",
        )

        ax.set_xticks(np.arange(-0.5, len(self.control_modes), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=2.2)
        ax.tick_params(which="minor", bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        if len(selected_norms) > 1:
            for boundary in range(len(selected_norms), len(rows), len(selected_norms)):
                ax.axhline(boundary - 0.5, color="#64748b", linewidth=1.8)

        for row in range(len(rows)):
            for col in range(len(self.control_modes)):
                if status[row][col] == "solved":
                    cell_value = values[row, col]
                    color_position = (cell_value - color_min) / max(
                        color_max - color_min, 1e-12
                    )
                    color = "white" if color_position >= 0.58 else "#111827"
                    ax.text(
                        col,
                        row,
                        f"{cell_value:.3f}%",
                        ha="center",
                        va="center",
                        color=color,
                        weight="bold",
                        fontsize=7.5,
                    )
                elif status[row][col] == "failed":
                    ax.text(
                        col,
                        row,
                        "FAIL",
                        ha="center",
                        va="center",
                        color="#374151",
                        weight="bold",
                        fontsize=9,
                    )
                else:
                    ax.text(
                        col,
                        row,
                        "N/A",
                        ha="center",
                        va="center",
                        color="#6b7280",
                        fontsize=11,
                    )

        colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
        colorbar.set_label(
            "Grid-driven curtailment (%)",
            rotation=90,
            labelpad=12,
            fontsize=10,
        )
        colorbar.ax.tick_params(labelsize=9)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
