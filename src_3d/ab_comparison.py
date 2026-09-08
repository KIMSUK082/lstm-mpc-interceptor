"""A/B study: acceleration-prediction MPC vs position-prediction MPC.

Runs the same scenario over many seeds, caches one pickle per seed, and
draws the comparison figures from them.

Self-contained on purpose: it defines its own scenario and never imports
main.py, so the single-run demo and this experiment can change
independently.  The scenario is written into every output, so a drift
between the two files shows up in the report rather than silently.

Three stages:

    sweep     run the seeds in parallel, one pickle each (cached - a seed
              that already has a pickle is not recomputed)
    timing    re-run a few seeds single-process, because a solve time
              measured under eight competing workers is inflated ~4x
    figures   read the pickles and draw G1-G5

    python ab_comparison.py                     all three
    python ab_comparison.py --skip-sweep        redraw from the cached pickles
    python ab_comparison.py --count 50          use fifty seeds
"""

import os

## Thread pinning has to happen before numpy or torch is imported, both for
## the workers and for the single-process timing stage.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
import numpy as np
import torch
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from missile import Missile
from model import AccelerationTargetPredictor, PositionTargetPredictor
from mpc import AccelerationDisturbanceMPC, PositionInterceptionMPC
from sim import ComparisonSim, load_comparison_result, save_comparison_result
from vehicle import Vehicle

torch.set_num_threads(1)


## ------------------------------------------------------------------ scenario

SCENARIO = {
    ## _sym: both methods bound each control axis at u_max / sqrt(2), the
    ## square inscribed in the 20 g circle.  Earlier runs under the name
    ## "aggressive_v300" let method B use u_max per axis and clipped the
    ## resultant afterwards, which gave B 41 % more single-axis authority
    ## than A.  Those pickles are kept for the before/after comparison.
    "name": "aggressive_v300_sym",
    "dt": 0.05,
    "T_max": 45.0,
    "observation_time": 2.0,
    "intercept_radius": 3.0,
    "maneuver_profile": "aggressive",
    "target_speed": 150.0,
    "pursuer_speed": 300.0,
    "pursuer_max_g": 20.0,
    "target_start": (-2000.0, -1500.0, 1200.0),
    "target_flight_path_angle_deg": 5.0,
    "target_heading_deg": 80.0,
    "launch_point": (0.0, 0.0, 300.0),
}

## Seeds 0-999 trained both LSTMs, so evaluation must start above them.
FIRST_EVALUATION_SEED = 1000

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = PROJECT_ROOT / "results_3d"
SEED_DIR = RESULT_DIR / "seed_runs"
FIGURE_DIR = RESULT_DIR / "presentation"
TIMING_PATH = RESULT_DIR / "timing.json"
METRICS_PATH = RESULT_DIR / "ab_metrics.csv"
REPORT_PATH = RESULT_DIR / "AB_REPORT.md"

## The engagement behind the two single-run figures: both methods intercept,
## the flight is long enough to read as a time series, and B spends half of it
## against the acceleration limit.
EXAMPLE_SEED = 10011

## Defaults for a run.  Edit these to change the study size; the matching
## command-line flags override them for a one-off run.
START_SEED = 10000
SEED_COUNT = 30
WORKERS = 8
TIMING_COUNT = 5


def seed_path(seed):
    return SEED_DIR / f"ab_{SCENARIO['name']}_seed{seed}.pkl"


def build_target():
    x, y, z = SCENARIO["target_start"]

    return Vehicle(
        x=x,
        y=y,
        z=z,
        v=SCENARIO["target_speed"],
        flight_path_angle=np.radians(SCENARIO["target_flight_path_angle_deg"]),
        heading=np.radians(SCENARIO["target_heading_deg"]),
    )


def build_missile(target):
    """Launch from the fixed point, pointed straight at the target."""
    launch = np.array(SCENARIO["launch_point"], dtype=float)
    relative = target.get_state()[0:3] - launch

    return Missile(
        x=launch[0],
        y=launch[1],
        z=launch[2],
        v=SCENARIO["pursuer_speed"],
        flight_path_angle=np.arctan2(
            relative[2],
            np.hypot(relative[0], relative[1]),
        ),
        heading=np.arctan2(relative[1], relative[0]),
        max_g=SCENARIO["pursuer_max_g"],
    )


def run_comparison(seed):
    """One seed: both methods flown against a single shared target path.

    Pure - it returns the result and writes nothing.
    """
    dt = SCENARIO["dt"]
    radius = SCENARIO["intercept_radius"]
    max_g = SCENARIO["pursuer_max_g"]

    target = build_target()

    acceleration_mpc = AccelerationDisturbanceMPC(
        dt=dt,
        horizon=20,
        disturbance_steps=8,
        max_g=max_g,
    )
    position_mpc = PositionInterceptionMPC(
        dt=dt,
        max_g=max_g,
        intercept_radius=radius,
        max_horizon=100,
        control_block_size=5,
    )

    simulation = ComparisonSim(
        target=target,
        acceleration_missile=build_missile(target),
        position_missile=build_missile(target),
        acceleration_predictor=AccelerationTargetPredictor(),
        position_predictor=PositionTargetPredictor(),
        acceleration_mpc=acceleration_mpc,
        position_mpc=position_mpc,
        dt=dt,
        T_max=SCENARIO["T_max"],
        seed=seed,
        observation_time=SCENARIO["observation_time"],
        intercept_radius=radius,
        maneuver_profile=SCENARIO["maneuver_profile"],
    )

    result = simulation.simulation()

    result["seed"] = seed
    result["scenario"] = dict(SCENARIO)
    result["acceleration"]["solver_failures"] = acceleration_mpc.solver_failures
    result["position"]["solver_failures"] = position_mpc.solver_failures
    result["position"]["qp_solve_count"] = position_mpc.qp_solve_count

    return result


## -------------------------------------------------------------- stage: sweep


def evaluate_seed(seed, recompute):
    path = seed_path(seed)

    if path.exists() and not recompute:
        return seed

    save_comparison_result(run_comparison(seed), path)

    return seed


def sweep(seeds, workers, recompute):
    SEED_DIR.mkdir(parents=True, exist_ok=True)
    pending = [seed for seed in seeds if recompute or not seed_path(seed).exists()]

    print(f"[sweep] {len(seeds)} seeds, {len(pending)} to compute")

    if not pending:
        return

    failed = []

    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(evaluate_seed, seed, recompute): seed for seed in pending
        }

        progress = tqdm(
            as_completed(futures),
            total=len(pending),
            desc="[sweep] seeds",
            unit="seed",
        )

        for future in progress:
            seed = futures[future]

            ## One unsolvable seed must not throw away the rest of the sweep.
            try:
                future.result()
            except Exception as error:
                failed.append((seed, error))
                ## tqdm.write keeps the bar intact instead of tearing it.
                progress.write(
                    f"[sweep] seed {seed} FAILED - "
                    f"{type(error).__name__}: {error}"
                )

            progress.set_postfix_str(f"seed {seed}")

        progress.close()

    if failed:
        print(f"[sweep] {len(failed)} seed(s) failed and are excluded:")

        for seed, error in failed:
            print(f"[sweep]   seed {seed}: {error}")


## ------------------------------------------------------------- stage: timing


def timing(seeds):
    """Solve time measured one process at a time.

    The sweep runs eight workers over twelve cores, which inflates every
    per-step time roughly fourfold.  G1 quotes this stage instead.
    """
    rows = []

    print(f"[timing] {len(seeds)} seeds, single process")

    progress = tqdm(seeds, desc="[timing] seeds", unit="seed")

    for seed in progress:
        result = run_comparison(seed)
        entry = {"seed": seed}

        for key in ("acceleration", "position"):
            method = result[key]
            steps = len(method["control_history"])
            entry[key] = {
                "steps": int(steps),
                "ms_per_step": 1000.0 * float(method["runtime"]) / max(steps, 1),
            }

        rows.append(entry)
        progress.write(
            f"[timing] seed {seed}: "
            f"A {entry['acceleration']['ms_per_step']:6.2f} ms/step | "
            f"B {entry['position']['ms_per_step']:6.2f} ms/step"
        )

    summary = {
        "scenario": dict(SCENARIO),
        "seeds": [row["seed"] for row in rows],
        "rows": rows,
    }

    for key in ("acceleration", "position"):
        values = [row[key]["ms_per_step"] for row in rows]
        summary[f"{key}_ms_per_step_median"] = float(np.median(values))
        summary[f"{key}_ms_per_step_max"] = float(np.max(values))

    summary["ratio"] = (
        summary["position_ms_per_step_median"]
        / summary["acceleration_ms_per_step_median"]
    )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    TIMING_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[timing] -> {TIMING_PATH}")

    return summary


## ---------------------------------------------------------------- figures


LABELS = {
    "acceleration": "A. Acceleration prediction",
    "position": "B. Position prediction",
}
SHORT = {"acceleration": "A", "position": "B"}
METHODS = ("acceleration", "position")

## Validated categorical slots 1 and 2 on a light surface: worst-pair CVD
## separation 24.7, well clear of the 8.0 floor.
COLORS = {"acceleration": "#2a78d6", "position": "#eb6834"}
MARKERS = {"acceleration": "o", "position": "s"}

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
INK_MUTED = "#8a8984"
TRUTH = "#3d3d3a"

G = 9.81
COLUMN_INCHES = 10.87  # 27.6 cm


def apply_style():
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": INK_MUTED,
            "axes.labelcolor": INK_SOFT,
            "axes.titlecolor": INK,
            "axes.linewidth": 1.0,
            "axes.grid": True,
            "grid.color": INK_MUTED,
            "grid.alpha": 0.20,
            "grid.linewidth": 0.9,
            "xtick.color": INK_SOFT,
            "ytick.color": INK_SOFT,
            "text.color": INK,
            "legend.frameon": False,
            "font.size": 20,
            "axes.titlesize": 23,
            "axes.labelsize": 20,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 19,
            "lines.linewidth": 3.0,
            "lines.markersize": 11,
        }
    )


def recess(axis):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.set_axisbelow(True)


def save_png(figure, directory, name):
    """Create the output directory if needed and write one PNG."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"  {path}")

    return path


## ------------------------------------------------------------------ metrics


def prediction_errors(result, key):
    """Forecast entry j made at step k describes step k + j + 1."""
    truth = np.asarray(result["target_trajectory"], dtype=float)[:, 0:3]
    predictions = result[key]["predictions"]
    horizon = len(predictions[0])
    errors = np.full((len(predictions), horizon), np.nan)

    for step, prediction in enumerate(predictions):
        if prediction is None:
            continue

        available = min(horizon, len(truth) - step - 1)

        if available <= 0:
            continue

        errors[step, :available] = np.linalg.norm(
            np.asarray(prediction[:available], dtype=float)
            - truth[step + 1 : step + 1 + available],
            axis=1,
        )

    return errors


def outcome_table(results):
    table = {}

    for key in METHODS:
        intercepted = [r[key]["intercepted"] for r in results.values()]
        times = [
            r[key]["intercept_time"]
            for r in results.values()
            if r[key]["intercept_time"] is not None
        ]
        misses = [r[key]["minimum_distance"] for r in results.values()]
        table[key] = {
            "success_rate": 100.0 * float(np.mean(intercepted)),
            "intercept_time": float(np.median(times)) if times else np.nan,
            "miss_distance": float(np.median(misses)),
        }

    return table


def method_rows(results, scenario):
    dt = scenario["dt"]
    limit = scenario["pursuer_max_g"] * G
    rows = []

    for seed, result in sorted(results.items()):
        for key in METHODS:
            method = result[key]
            controls = np.asarray(method["control_history"], dtype=float)
            magnitude = (
                np.linalg.norm(controls, axis=1) if len(controls) else np.zeros(1)
            )
            rows.append(
                {
                    "seed": seed,
                    "method": key,
                    "intercepted": int(method["intercepted"]),
                    "intercept_time_s": (
                        ""
                        if method["intercept_time"] is None
                        else float(method["intercept_time"])
                    ),
                    "minimum_distance_m": float(method["minimum_distance"]),
                    "mean_acceleration_g": float(np.mean(magnitude) / G),
                    "peak_acceleration_g": float(np.max(magnitude) / G),
                    "saturation_ratio": float(np.mean(magnitude >= 0.99 * limit)),
                    "delta_v_ms": float(np.sum(magnitude) * dt),
                    "solver_failures": int(method.get("solver_failures", 0)),
                }
            )

    return rows


## ------------------------------------------------------------------ figures


def figure_computation(timing_summary, directory):
    values = [timing_summary[f"{key}_ms_per_step_median"] for key in METHODS]

    figure, axis = plt.subplots(figsize=(COLUMN_INCHES, 3.6))
    bars = axis.barh(
        [SHORT[key] for key in METHODS],
        values,
        color=[COLORS[key] for key in METHODS],
        height=0.55,
    )
    axis.invert_yaxis()

    for bar, value in zip(bars, values):
        axis.text(
            value + max(values) * 0.02,
            bar.get_y() + bar.get_height() / 2.0,
            f"{value:.1f} ms",
            va="center",
            fontsize=22,
            fontweight="bold",
            color=INK,
        )

    axis.set_xlim(0.0, max(values) * 1.28)
    axis.set_xlabel("Solve time per control step [ms]")
    axis.set_title("Computation per control step", loc="left")
    axis.annotate(
        f"B needs {values[1] / values[0]:.1f}x the computation of A",
        xy=(0.99, 0.80),
        xycoords="axes fraction",
        ha="right",
        va="center",
        fontsize=20,
        color=INK_SOFT,
    )
    axis.grid(axis="y", visible=False)
    recess(axis)
    figure.tight_layout()

    return save_png(figure, directory, "G1_computation.png")


def figure_control_effort(result, seed, scenario, directory):
    dt = scenario["dt"]
    limit = scenario["pursuer_max_g"]
    figure, axes = plt.subplots(2, 1, figsize=(COLUMN_INCHES, 7.6), sharex=True)

    for key in METHODS:
        controls = np.asarray(result[key]["control_history"], dtype=float)
        magnitude = np.linalg.norm(controls, axis=1) / G
        time = np.arange(len(magnitude)) * dt

        axes[0].plot(time, magnitude, color=COLORS[key], linewidth=2.6)
        axes[1].plot(
            time,
            np.cumsum(magnitude * G) * dt,
            color=COLORS[key],
            linewidth=2.6,
        )

        total = float(np.sum(magnitude * G) * dt)
        axes[1].annotate(
            f"{total:.0f} m/s",
            xy=(time[-1], total),
            xytext=(-8, -18 if key == "acceleration" else 12),
            textcoords="offset points",
            ha="right",
            fontsize=20,
            fontweight="bold",
            color=COLORS[key],
        )

    axes[0].axhline(limit, color=INK_MUTED, linestyle="--", linewidth=1.8)
    axes[0].annotate(
        f"{limit:g} g interceptor limit",
        xy=(0.01, limit),
        xycoords=("axes fraction", "data"),
        xytext=(0, 6),
        textcoords="offset points",
        ha="left",
        fontsize=18,
        color=INK_SOFT,
    )
    axes[0].set_ylim(0.0, limit * 1.22)
    axes[0].set_ylabel("Commanded\nacceleration [g]")
    axes[0].set_title(f"Control input over one engagement  (seed {seed})", loc="left")
    axes[1].set_ylabel("Cumulative\ncontrol effort [m/s]")
    axes[1].set_xlabel("Time [s]")

    for axis in axes:
        recess(axis)

    figure.legend(
        handles=[
            plt.Line2D([], [], color=COLORS[key], linewidth=3.0, label=LABELS[key])
            for key in METHODS
        ],
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.02),
    )
    figure.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))

    return save_png(figure, directory, "G2_control_effort.png")


def pick_prediction_step(result):
    """A moment where the target is actually manoeuvring, mid-flight.

    A method stops forecasting once it has intercepted, so its later
    entries are None.  Only steps where both methods still predict can be
    compared.
    """
    commands = np.asarray(result["target_control_history"], dtype=float)
    acceleration = result["acceleration"]["predictions"]
    position = result["position"]["predictions"]
    horizon = len(acceleration[0])
    limit = min(
        len(acceleration),
        len(position),
        len(commands),
        len(result["target_trajectory"]) - horizon - 1,
    )

    usable = [
        step
        for step in range(limit)
        if acceleration[step] is not None and position[step] is not None
    ]

    if not usable:
        raise ValueError(
            "No step has a forecast from both methods."
        )

    ## Skip the opening dive, where the target has barely manoeuvred yet.
    candidates = usable[int(0.25 * len(usable)) :] or usable
    lateral = np.linalg.norm(commands[candidates, 1:3], axis=1)

    return candidates[int(np.argmax(lateral))]


def figure_prediction_example(result, seed, scenario, directory):
    dt = scenario["dt"]
    step = pick_prediction_step(result)
    horizon = len(result["acceleration"]["predictions"][step])
    lookahead = (np.arange(horizon) + 1) * dt

    truth = np.asarray(result["target_trajectory"], dtype=float)[
        step + 1 : step + 1 + horizon, 0:3
    ]
    predicted = {
        key: np.asarray(result[key]["predictions"][step], dtype=float)[:horizon]
        for key in METHODS
    }

    figure, axes = plt.subplots(3, 1, figsize=(COLUMN_INCHES, 10.2), sharex=True)

    for index, name in enumerate(("X", "Y", "Z")):
        axis = axes[index]
        ## A tracks the truth closely enough to vanish under a thin line, so
        ## the truth is drawn as a wide band underneath instead.
        axis.plot(
            lookahead,
            truth[:, index],
            color=TRUTH,
            linewidth=8.0,
            alpha=0.28,
            solid_capstyle="round",
            label="Ground truth",
            zorder=1,
        )

        for key in METHODS:
            axis.plot(
                lookahead,
                predicted[key][:, index],
                color=COLORS[key],
                linewidth=2.8,
                linestyle="--",
                marker=MARKERS[key],
                markersize=11,
                markeredgecolor=SURFACE,
                markeredgewidth=1.4,
                label=LABELS[key],
                zorder=3,
            )

        axis.set_ylabel(f"{name} [m]")
        recess(axis)

    axes[0].set_title(
        f"Predicted vs true target position  " f"(seed {seed}, t = {step * dt:.2f} s)",
        loc="left",
    )
    axes[2].set_xlabel("Prediction horizon [s]")

    handles, texts = axes[0].get_legend_handles_labels()
    figure.legend(
        handles=handles,
        labels=texts,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.015),
        fontsize=18,
    )
    figure.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))

    return save_png(figure, directory, "G3_prediction_example.png")


def figure_prediction_error(results, scenario, directory):
    dt = scenario["dt"]
    stacked = {
        key: [prediction_errors(r, key) for r in results.values()] for key in METHODS
    }
    horizon = min(min(errors.shape[1] for errors in stacked[key]) for key in stacked)

    figure, axis = plt.subplots(figsize=(COLUMN_INCHES, 5.4))
    lookahead = (np.arange(horizon) + 1) * dt

    for key in METHODS:
        errors = np.vstack([e[:, :horizon] for e in stacked[key]])
        median = np.nanmedian(errors, axis=0)

        axis.fill_between(
            lookahead,
            np.nanpercentile(errors, 25, axis=0),
            np.nanpercentile(errors, 75, axis=0),
            color=COLORS[key],
            alpha=0.16,
            linewidth=0,
        )
        axis.plot(
            lookahead,
            median,
            color=COLORS[key],
            marker=MARKERS[key],
            markersize=11,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
        )
        axis.annotate(
            SHORT[key],
            xy=(lookahead[0], median[0]),
            xytext=(-14, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=24,
            fontweight="bold",
            color=COLORS[key],
        )
        axis.annotate(
            f"{median[-1]:.2f} m",
            xy=(lookahead[-1], median[-1]),
            xytext=(-6, 12),
            textcoords="offset points",
            ha="right",
            fontsize=20,
            fontweight="bold",
            color=COLORS[key],
        )

    ## Without this reference, "5.9 m" carries no sense of scale.
    step_travel = scenario["target_speed"] * dt
    axis.axhline(step_travel, color=INK_MUTED, linestyle=":", linewidth=2.0)
    axis.annotate(
        f"{step_travel:.1f} m = distance the target covers in one step",
        xy=(0.99, step_travel),
        xycoords=("axes fraction", "data"),
        xytext=(0, -26),
        textcoords="offset points",
        ha="right",
        fontsize=18,
        color=INK_SOFT,
    )

    axis.set_xlabel("Prediction horizon [s]")
    axis.set_ylabel("Position error [m]")
    axis.set_title(
        f"Prediction error, median and IQR over {len(results)} seeds",
        loc="left",
    )
    axis.margins(x=0.09)
    recess(axis)
    figure.tight_layout()

    return save_png(figure, directory, "G4_prediction_error.png")


def figure_outcome(results, scenario, directory):
    radius = scenario["intercept_radius"]
    table = outcome_table(results)

    panels = [
        ("Interception success [%]", "success_rate", "{:.0f}%", None),
        ("Time to intercept [s]  (median)", "intercept_time", "{:.1f} s", None),
        ("Miss distance [m]  (median)", "miss_distance", "{:.2f} m", radius),
    ]

    figure, axes = plt.subplots(3, 1, figsize=(COLUMN_INCHES, 8.0))

    for axis, (title, field, template, reference) in zip(axes, panels):
        values = [table[key][field] for key in METHODS]
        bars = axis.barh(
            [SHORT[key] for key in METHODS],
            values,
            color=[COLORS[key] for key in METHODS],
            height=0.55,
        )
        axis.invert_yaxis()
        span = max(values)

        for bar, value in zip(bars, values):
            axis.text(
                value + span * 0.02,
                bar.get_y() + bar.get_height() / 2.0,
                template.format(value),
                va="center",
                fontsize=21,
                fontweight="bold",
                color=INK,
            )

        upper = span * 1.30

        if reference is not None:
            upper = max(upper, reference * 1.18)
            axis.axvline(reference, color=INK_MUTED, linestyle="--", linewidth=1.8)
            axis.annotate(
                f"{reference:g} m hit radius",
                xy=(reference, 0.06),
                xycoords=("data", "axes fraction"),
                xytext=(8, 0),
                textcoords="offset points",
                fontsize=17,
                color=INK_SOFT,
                va="bottom",
            )

        axis.set_xlim(0.0, upper)
        axis.set_title(title, loc="left", fontsize=21)
        axis.grid(axis="y", visible=False)
        recess(axis)

    figure.suptitle(
        f"Interception outcome over {len(results)} unseen seeds",
        fontsize=24,
        fontweight="bold",
        x=0.01,
        ha="left",
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

    return save_png(figure, directory, "G5_outcome.png")


## ------------------------------------------------------------------- tables


def write_metrics_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"  {path}")

    return path


def write_report(results, rows, timing_summary, scenario, path):
    table = outcome_table(results)

    def mean_of(key, field):
        values = [
            row[field] for row in rows if row["method"] == key and row[field] != ""
        ]

        return float(np.mean(values))

    lines = [
        "# A/B study",
        "",
        f"- Seeds: {len(results)}",
        f"- Scenario `{scenario['name']}`: target "
        f"{scenario['target_speed']:.0f} m/s, both pursuers "
        f"{scenario['pursuer_speed']:.0f} m/s, hit radius "
        f"{scenario['intercept_radius']:g} m, profile "
        f"`{scenario['maneuver_profile']}`",
        "- Solve times come from the single-process timing stage, not from "
        "the parallel sweep.",
        "",
        "| Metric | A. Acceleration | B. Position |",
        "|---|---:|---:|",
        f"| Success rate | {table['acceleration']['success_rate']:.1f}% "
        f"| {table['position']['success_rate']:.1f}% |",
        f"| Median intercept time "
        f"| {table['acceleration']['intercept_time']:.2f} s "
        f"| {table['position']['intercept_time']:.2f} s |",
        f"| Median miss distance "
        f"| {table['acceleration']['miss_distance']:.3f} m "
        f"| {table['position']['miss_distance']:.3f} m |",
        f"| Mean commanded acceleration "
        f"| {mean_of('acceleration', 'mean_acceleration_g'):.2f} g "
        f"| {mean_of('position', 'mean_acceleration_g'):.2f} g |",
        f"| Steps at the acceleration limit "
        f"| {100.0 * mean_of('acceleration', 'saturation_ratio'):.2f}% "
        f"| {100.0 * mean_of('position', 'saturation_ratio'):.2f}% |",
        f"| Cumulative delta-v "
        f"| {mean_of('acceleration', 'delta_v_ms'):.1f} m/s "
        f"| {mean_of('position', 'delta_v_ms'):.1f} m/s |",
        f"| Solve time per step "
        f"| {timing_summary['acceleration_ms_per_step_median']:.2f} ms "
        f"| {timing_summary['position_ms_per_step_median']:.2f} ms |",
    ]

    ## JSON turns the tuples in the scenario into lists, so compare both
    ## sides after the same round trip.
    ## Both sides go through the same round trip: a summary just produced in
    ## memory still holds tuples, while one read back from the file holds
    ## lists, and those must not read as a drift.
    recorded = json.loads(json.dumps(timing_summary.get("scenario")))
    current = json.loads(json.dumps(scenario))

    if recorded != current:
        lines.extend(
            [
                "",
                "> WARNING: the timing file was produced under a different "
                "scenario than this run. Re-run the timing stage.",
            ]
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  {path}")

    return path


## -------------------------------------------------------------------- entry


def render_all(
    results,
    example_seed,
    timing_summary,
    scenario,
    figure_dir,
    metrics_path,
    report_path,
):
    """Draw G1-G5 and write the two tables."""
    apply_style()
    example = results[example_seed]

    figure_computation(timing_summary, figure_dir)
    figure_control_effort(example, example_seed, scenario, figure_dir)
    figure_prediction_example(example, example_seed, scenario, figure_dir)
    figure_prediction_error(results, scenario, figure_dir)
    figure_outcome(results, scenario, figure_dir)

    rows = method_rows(results, scenario)
    write_metrics_csv(rows, metrics_path)
    write_report(results, rows, timing_summary, scenario, report_path)


## --------------------------------------------------------------------- entry


def load_results(seeds):
    results = {}

    for seed in seeds:
        path = seed_path(seed)

        if path.exists():
            results[seed] = load_comparison_result(path)

    if not results:
        raise SystemExit("No cached seed results. Run without --skip-sweep first.")

    return results


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-seed", type=int, default=START_SEED)
    parser.add_argument("--count", type=int, default=SEED_COUNT)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument(
        "--timing-count",
        type=int,
        default=TIMING_COUNT,
        help="Seeds re-run single-process for the solve-time figure.",
    )
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--skip-sweep", action="store_true")
    parser.add_argument(
        "--skip-timing",
        action="store_true",
        help="Reuse the existing timing file.",
    )

    return parser.parse_args()


def main():
    arguments = parse_arguments()

    if arguments.start_seed < FIRST_EVALUATION_SEED:
        raise SystemExit(
            f"Evaluation seeds must start at or above "
            f"{FIRST_EVALUATION_SEED}; seeds below that trained the LSTMs."
        )

    seeds = list(range(arguments.start_seed, arguments.start_seed + arguments.count))

    if not arguments.skip_sweep:
        sweep(seeds, arguments.workers, arguments.recompute)

    if arguments.skip_timing:
        if not TIMING_PATH.exists():
            raise SystemExit(f"--skip-timing needs an existing {TIMING_PATH.name}.")
        timing_summary = json.loads(TIMING_PATH.read_text(encoding="utf-8"))
    else:
        timing_summary = timing(seeds[: arguments.timing_count])

    results = load_results(seeds)
    example_seed = EXAMPLE_SEED

    if example_seed not in results:
        example_seed = min(results)
        print(
            f"[figures] seed {EXAMPLE_SEED} is not in this range; "
            f"using seed {example_seed} for the single-run figures."
        )

    print(f"[figures] {len(results)} seeds loaded")

    render_all(
        results=results,
        example_seed=example_seed,
        timing_summary=timing_summary,
        scenario=SCENARIO,
        figure_dir=FIGURE_DIR,
        metrics_path=METRICS_PATH,
        report_path=REPORT_PATH,
    )


if __name__ == "__main__":
    main()
