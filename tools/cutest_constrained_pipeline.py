#!/usr/bin/env python3
"""PyCUTEst constrained benchmark for NOMAD ordering strategies.

This script mirrors the More-Wild workflow for a small constrained CUTEst set:

1. Run PyNomad on selected constrained PyCUTEst problems.
2. Format raw stats into RunnerPost's STRATEGY/PROBLEM/stats.instance.txt layout.
3. Synchronize initial objective values and generate RunnerPost .def files.
4. Optionally run RunnerPost and compile the profile figures.

For constrained omniscient sorting, CUTEst constraints are converted to NOMAD
PB constraints g(x) <= 0. The true violation is

    h(x) = sum_j max(g_j(x), 0)^2.

OMNISCIENT returns the true objective and true PB constraints to NOMAD's
SURROGATE ordering. REVERSE_OMNI returns an artificial surrogate with
constraint violation decreasing in true h and objective -f, which reverses
the lexicographic (h, f) priority used by NOMAD's constrained ordering.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import json
import math
import multiprocessing as mp
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from mw_runnerpost_pipeline import compile_tex, run_runnerpost, style_tex_files


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


DEFAULT_PROBLEMS = [
    "HS21", "HS22", "HS23", "HS24", "HS26", "HS27", "HS28", "HS29", "HS30",
    "HS31", "HS32", "HS33", "HS34", "HS35", "HS36", "HS37", "HS43", "HS44",
    "HS46", "HS47", "HS48", "HS49", "HS50", "HS51", "HS52", "HS53", "HS54",
    "HS55", "HS56", "HS57", "HS60", "HS61", "HS62", "HS63", "HS64", "HS65",
    "HS66", "HS67", "HS68", "HS69", "HS70", "HS71", "HS72", "HS73", "HS74",
    "HS75", "HS76", "HS77", "HS78", "HS79", "HS80", "HS81", "HS83", "HS84",
    "HS85", "HS86", "HS87", "HS88", "HS89", "HS90", "HS91", "HS93", "HS95",
    "HS96", "HS97", "HS98", "HS99", "HS100", "HS106", "HS107", "HS108",
    "HS109", "HS110", "HS111", "HS112", "HS113", "HS114", "HS116", "HS117",
    "HS118", "HS119",
]

STRATEGIES: dict[str, str] = {
    "DIR_LAST_SUCCESS": "DIR_LAST_SUCCESS",
    "QUADRATIC": "QUADRATIC_MODEL",
    "RANDOM": "RANDOM",
    "LEXICO": "LEXICOGRAPHICAL",
    "OMNISCIENT": "SURROGATE",
    "REVERSE_OMNI": "SURROGATE",
}

DEFAULT_STRATEGY_ORDER = [
    "QUADRATIC",
    "RANDOM",
    "LEXICO",
    "DIR_LAST_SUCCESS",
    "OMNISCIENT",
    "REVERSE_OMNI",
]

SURROGATE_STRATEGIES = {"OMNISCIENT", "REVERSE_OMNI"}


@dataclasses.dataclass(frozen=True)
class StatsPoint:
    eval_count: int
    objective: float


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def parse_problem_list(spec: str) -> list[str]:
    return [part.strip().upper() for part in spec.split(",") if part.strip()]


def finite_bound(value: float, fallback: float) -> float:
    return float(value) if math.isfinite(float(value)) else fallback


def import_cutest_problem(name: str):
    import pycutest

    return pycutest.import_problem(name)


def available_cutest_problem_names() -> list[str]:
    import pycutest

    try:
        return sorted(str(name).upper() for name in pycutest.find_problems())
    except Exception:
        return DEFAULT_PROBLEMS


def variable_bounds_to_pb(problem, x: list[float] | np.ndarray) -> list[float]:
    """Convert finite variable bounds bl <= x <= bu to PB values g(x) <= 0."""
    x_array = np.asarray(x, dtype=float)
    values: list[float] = []
    for x_value, lower, upper in zip(x_array, getattr(problem, "bl", []), getattr(problem, "bu", [])):
        x_float = float(x_value)
        if math.isfinite(float(lower)):
            values.append(float(lower) - x_float)
        if math.isfinite(float(upper)):
            values.append(x_float - float(upper))
    return values


def nonlinear_constraints_to_pb(problem, x: list[float] | np.ndarray) -> list[float]:
    """Convert CUTEst nonlinear bounds cl <= c(x) <= cu to PB values g(x) <= 0."""
    if getattr(problem, "m", 0) == 0:
        return []
    x_array = np.asarray(x, dtype=float)
    cons = np.atleast_1d(problem.cons(x_array))
    values: list[float] = []
    for c_value, lower, upper in zip(cons, problem.cl, problem.cu):
        c_float = float(c_value)
        if math.isfinite(float(lower)):
            values.append(float(lower) - c_float)
        if math.isfinite(float(upper)):
            values.append(c_float - float(upper))
    return values


def cutest_constraints_to_pb(problem, x: list[float] | np.ndarray) -> list[float]:
    """Convert all finite CUTEst bounds/constraints to NOMAD PB values g(x) <= 0."""
    return variable_bounds_to_pb(problem, x) + nonlinear_constraints_to_pb(problem, x)


def pb_violation(pb_values: Iterable[float]) -> float:
    return sum(max(value, 0.0) ** 2 for value in pb_values)


def bound_sampling_box(problem, x0: list[float], radius: float) -> tuple[np.ndarray, np.ndarray]:
    x0_array = np.asarray(x0, dtype=float)
    lower = np.empty_like(x0_array)
    upper = np.empty_like(x0_array)
    for i, x0_value in enumerate(x0_array):
        scale = max(1.0, abs(float(x0_value)))
        fallback_lower = float(x0_value) - radius * scale
        fallback_upper = float(x0_value) + radius * scale
        bl = float(getattr(problem, "bl", [])[i])
        bu = float(getattr(problem, "bu", [])[i])
        lower[i] = bl if math.isfinite(bl) else fallback_lower
        upper[i] = bu if math.isfinite(bu) else fallback_upper
        if lower[i] > upper[i]:
            lower[i], upper[i] = upper[i], lower[i]
        if lower[i] == upper[i]:
            lower[i] -= 1.0
            upper[i] += 1.0
    return lower, upper


def unique_start(starts: list[list[float]], candidate: list[float], tol: float = 1e-10) -> bool:
    candidate_array = np.asarray(candidate, dtype=float)
    return all(np.linalg.norm(candidate_array - np.asarray(start, dtype=float), ord=np.inf) > tol for start in starts)


def generate_initial_starts(
    problem,
    *,
    n_starts: int,
    mode: str,
    seed: int,
    candidate_count: int,
    feasibility_tol: float,
    radius: float,
) -> tuple[list[list[float]], list[dict]]:
    """Generate CUTEst starts, prioritizing feasible points when requested."""
    x0 = [float(value) for value in np.asarray(problem.x0, dtype=float)]
    starts: list[list[float]] = []
    records: list[dict] = []

    def add_start(candidate: list[float], source: str) -> None:
        objective, _pb_values, h_value = evaluate_cutest(problem, candidate)
        starts.append(candidate)
        records.append({"source": source, "h": h_value, "f": objective})

    x0_h = pb_violation(cutest_constraints_to_pb(problem, x0))
    if mode in {"cutest", "mixed"} or x0_h <= feasibility_tol:
        add_start(x0, "cutest_x0")

    if mode == "cutest":
        while len(starts) < n_starts:
            add_start(x0, "cutest_x0_repeat")
        return starts[:n_starts], records[:n_starts]

    rng = np.random.default_rng(seed)
    lower, upper = bound_sampling_box(problem, x0, radius)
    feasible_pool: list[tuple[float, float, list[float]]] = []
    infeasible_pool: list[tuple[float, float, list[float]]] = []

    for _ in range(candidate_count):
        candidate_array = rng.uniform(lower, upper)
        candidate = [float(value) for value in candidate_array]
        objective, _pb_values, h_value = evaluate_cutest(problem, candidate)
        if h_value <= feasibility_tol:
            feasible_pool.append((h_value, objective, candidate))
        else:
            infeasible_pool.append((h_value, objective, candidate))

    feasible_pool.sort(key=lambda item: (item[0], item[1]))
    infeasible_pool.sort(key=lambda item: (item[0], item[1]))

    for h_value, objective, candidate in feasible_pool:
        if len(starts) >= n_starts:
            break
        if unique_start(starts, candidate):
            starts.append(candidate)
            records.append({"source": "random_feasible", "h": h_value, "f": objective})

    # If the random search cannot find enough feasible points, keep the best
    # least-infeasible starts rather than silently reducing the benchmark size.
    for h_value, objective, candidate in infeasible_pool:
        if len(starts) >= n_starts:
            break
        if unique_start(starts, candidate):
            starts.append(candidate)
            records.append({"source": "least_infeasible_fallback", "h": h_value, "f": objective})

    while len(starts) < n_starts:
        add_start(x0, "cutest_x0_repeat")

    return starts[:n_starts], records[:n_starts]


def evaluate_cutest(problem, x: list[float]) -> tuple[float, list[float], float]:
    try:
        x_array = np.asarray(x, dtype=float)
        objective = float(problem.obj(x_array))
        pb_values = cutest_constraints_to_pb(problem, x_array)
        h_value = pb_violation(pb_values)
        if not math.isfinite(objective) or not math.isfinite(h_value):
            raise ValueError("non-finite CUTEst evaluation")
        return objective, pb_values, h_value
    except Exception:
        # Failed CUTEst evaluations are treated as highly infeasible.
        return 1e20, [1e10], 1e20


def bbo_string(objective: float, pb_values: list[float]) -> bytes:
    return " ".join([f"{objective:.17g}", *(f"{value:.17g}" for value in pb_values)]).encode("UTF-8")


def make_bb(problem_name: str, dim: int, pb_count: int):
    problem = import_cutest_problem(problem_name)

    def bb(opt_param):
        x = [opt_param.get_coord(i) for i in range(dim)]
        objective, pb_values, _h_value = evaluate_cutest(problem, x)
        if len(pb_values) != pb_count:
            pb_values = (pb_values + [1e10] * pb_count)[:pb_count]
        opt_param.setBBO(bbo_string(objective, pb_values))
        return 1

    return bb


def make_surrogate(problem_name: str, dim: int, pb_count: int, *, reverse: bool):
    problem = import_cutest_problem(problem_name)

    def surrogate(opt_param):
        x = [opt_param.get_coord(i) for i in range(dim)]
        objective, pb_values, h_value = evaluate_cutest(problem, x)

        if reverse:
            # NOMAD sorts by increasing surrogate h, then increasing surrogate f.
            # 1/(1+h) reverses the true h priority; -f reverses objective ties.
            reverse_pb = [1.0 / (1.0 + h_value)] + [0.0] * max(pb_count - 1, 0)
            opt_param.setBBO(bbo_string(-objective, reverse_pb))
        else:
            if len(pb_values) != pb_count:
                pb_values = (pb_values + [1e10] * pb_count)[:pb_count]
            opt_param.setBBO(bbo_string(objective, pb_values))
        return 1

    return surrogate


def read_stats_file(path: Path) -> list[StatsPoint]:
    rows: list[StatsPoint] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                rows.append(StatsPoint(int(parts[0]), float(parts[1])))
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"{path}: no numeric stats rows found")
    return rows


def write_stats_file(path: Path, rows: Iterable[StatsPoint]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(f"{row.eval_count} {row.objective:.17g}\n")


def last_stats_summary(path: Path) -> tuple[int | str, float | str]:
    try:
        rows = read_stats_file(path)
        return rows[-1].eval_count, rows[-1].objective
    except Exception:
        return "NA", "NA"


def run_task(task: tuple) -> str:
    (
        problem_name,
        sub_id,
        strategy,
        sort_type,
        dim,
        pb_count,
        x0,
        results_dir_s,
        max_bb_eval,
        seed,
    ) = task

    import PyNomad

    results_dir = Path(results_dir_s)
    run_dir = results_dir / f"{problem_name}_{sub_id}" / strategy
    run_dir.mkdir(parents=True, exist_ok=True)

    previous_cwd = Path.cwd()
    os.chdir(run_dir)
    try:
        Path("x0.txt").write_text(" ".join(f"{value:.17g}" for value in x0) + "\n", encoding="utf-8")

        target_points = min(200, max(10, 4 * dim))
        output_types = " ".join(["OBJ"] + ["PB"] * pb_count)
        params = [
            f"BB_OUTPUT_TYPE {output_types}",
            "DIRECTION_TYPE ORTHO 2N",
            f"TRIAL_POINT_MAX_ADD_UP {target_points}",
            f"MAX_BB_EVAL {max_bb_eval}",
            f"SEED {seed}",
            f"EVAL_QUEUE_SORT {sort_type}",
            "STATS_FILE stats.txt BBE OBJ CONS_H",
            "ADD_SEED_TO_FILE_NAMES no",
            "EVAL_OPPORTUNISTIC yes",
            "DISPLAY_DEGREE 0",
            "DISPLAY_ALL_EVAL no",
            "QUAD_MODEL_SEARCH no",
            "SGTELIB_MODEL_SEARCH no",
            "NM_SEARCH no",
            "SPECULATIVE_SEARCH no",
            "VNS_MADS_SEARCH no",
            "VNSMART_MADS_SEARCH no",
            "LH_SEARCH 0 0",
        ]

        bb = make_bb(problem_name, dim, pb_count)
        if strategy in SURROGATE_STRATEGIES:
            surrogate = make_surrogate(problem_name, dim, pb_count, reverse=(strategy == "REVERSE_OMNI"))
            PyNomad.optimize(bb, x0, [], [], params, surrogate)
        else:
            PyNomad.optimize(bb, x0, [], [], params)

        bbe, obj = last_stats_summary(Path("stats.txt"))
        return f"done {problem_name}_{sub_id} {strategy}: bbe={bbe} obj={obj}"
    except Exception as exc:
        return f"crash {problem_name}_{sub_id} {strategy}: {exc}"
    finally:
        os.chdir(previous_cwd)


def inspect_problem(args: argparse.Namespace, name: str) -> dict:
    problem = import_cutest_problem(name)
    x0_array = np.asarray(problem.x0, dtype=float)
    x0 = [float(value) for value in x0_array]
    pb_values = cutest_constraints_to_pb(problem, x0)
    starts, start_records = generate_initial_starts(
        problem,
        n_starts=args.n_subinstances,
        mode=args.start_mode,
        seed=args.start_seed,
        candidate_count=args.candidate_starts,
        feasibility_tol=args.feasibility_tol,
        radius=args.start_radius,
    )
    return {
        "name": name,
        "n": int(problem.n),
        "m_cutest": int(getattr(problem, "m", 0)),
        "m_pb": len(pb_values),
        "x0": x0,
        "x0_h": pb_violation(pb_values),
        "starts": starts,
        "start_records": start_records,
    }


def simulate(args: argparse.Namespace) -> None:
    results_dir = args.results_dir.resolve()
    if results_dir.exists() and args.clean:
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    problem_names = args.problems
    if args.auto_select_constrained:
        problem_names = [name for name in available_cutest_problem_names() if name.startswith(args.auto_prefix.upper())]

    metadata: dict[str, dict] = {}
    for problem_name in problem_names:
        if args.auto_select_constrained and len(metadata) >= args.auto_count:
            break
        try:
            info = inspect_problem(args, problem_name)
        except Exception as exc:
            if args.skip_missing:
                print(
                    f"skip unavailable CUTEst problem: {problem_name} ({type(exc).__name__}: {exc})",
                    flush=True,
                )
                continue
            raise
        if info["m_pb"] == 0:
            if args.skip_unconstrained:
                print(f"skip unconstrained CUTEst problem: {problem_name}", flush=True)
                continue
            raise ValueError(f"{problem_name}: no finite CUTEst constraints were found")
        if info["n"] < args.min_dim or info["n"] > args.max_dim:
            continue
        metadata[problem_name] = info

    if not metadata:
        raise ValueError("no constrained CUTEst problems selected")

    print("Selected CUTEst problems:", ", ".join(metadata), flush=True)
    for problem_name, info in metadata.items():
        n_feas = sum(1 for record in info["start_records"] if record["h"] <= args.feasibility_tol)
        print(
            f"  {problem_name}: n={info['n']} m_pb={info['m_pb']} "
            f"starts={len(info['starts'])} feasible_starts={n_feas}",
            flush=True,
        )

    (results_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    tasks = []
    for problem_name, info in metadata.items():
        for sub_id, x0 in enumerate(info["starts"], start=1):
            seed = sub_id * args.seed_stride
            for strategy, sort_type in STRATEGIES.items():
                tasks.append(
                    (
                        problem_name,
                        sub_id,
                        strategy,
                        sort_type,
                        info["n"],
                        info["m_pb"],
                        x0,
                        str(results_dir),
                        args.max_bb_eval,
                        seed,
                    )
                )

    ctx = mp.get_context(args.mp_context)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as executor:
        futures = [executor.submit(run_task, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            print(future.result(), flush=True)


def format_results(source_dir: Path, target_dir: Path, strategies: list[str]) -> int:
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True)

    metadata_path = source_dir / "metadata.json"
    if metadata_path.exists():
        shutil.copy(metadata_path, target_dir / "metadata.json")

    copied = 0
    for run_dir in sorted(path for path in source_dir.iterdir() if path.is_dir()):
        if "_" not in run_dir.name:
            continue
        problem_name, instance_id = run_dir.name.rsplit("_", 1)
        if not instance_id.isdigit():
            continue
        for strategy in strategies:
            stats_file = run_dir / strategy / "stats.txt"
            if not stats_file.exists() or stats_file.stat().st_size == 0:
                continue
            rows = read_stats_file(stats_file)
            write_stats_file(target_dir / strategy / problem_name / f"stats.{instance_id}.txt", rows)
            copied += 1
    return copied


def sync_results(source_dir: Path, sync_dir: Path, strategies: list[str], strict_complete: bool) -> tuple[int, int]:
    if sync_dir.exists():
        shutil.rmtree(sync_dir)
    sync_dir.mkdir(parents=True)

    metadata = json.loads((source_dir / "metadata.json").read_text(encoding="utf-8"))

    available: dict[str, set[tuple[str, str]]] = {}
    for strategy in strategies:
        strategy_dir = source_dir / strategy
        entries: set[tuple[str, str]] = set()
        for problem_dir in sorted(path for path in strategy_dir.iterdir() if path.is_dir()):
            for stats_file in problem_dir.glob("stats.*.txt"):
                parts = stats_file.name.split(".")
                if len(parts) >= 3:
                    entries.add((problem_dir.name, parts[1]))
        available[strategy] = entries

    common_entries = set.intersection(*(available[strategy] for strategy in strategies))
    if not common_entries:
        raise ValueError("no common problem instances found across selected strategies")
    if strict_complete:
        for strategy in strategies:
            missing = sorted(available[strategy].symmetric_difference(common_entries))
            if missing:
                raise ValueError(f"{strategy}: incomplete data set; examples={missing[:10]}")

    valid_problems: dict[str, list[str]] = {}
    written = 0
    for problem_name, instance_id in sorted(common_entries):
        rows_by_strategy = {
            strategy: read_stats_file(source_dir / strategy / problem_name / f"stats.{instance_id}.txt")
            for strategy in strategies
        }
        f0 = rows_by_strategy[strategies[0]][0].objective
        valid_problems.setdefault(problem_name, []).append(instance_id)
        for strategy in strategies:
            normalized = [StatsPoint(1, f0)]
            normalized.extend(row for row in rows_by_strategy[strategy] if row.eval_count != 1)
            write_stats_file(sync_dir / strategy / problem_name / f"stats.{instance_id}.txt", normalized)
            written += 1

    write_defs(sync_dir, strategies, valid_problems, metadata)
    return len(valid_problems), written


def write_defs(sync_dir: Path, strategies: list[str], valid_problems: dict[str, list[str]], metadata: dict) -> None:
    with (sync_dir / "problem_selection.def").open("w", encoding="utf-8") as handle:
        for problem_name in sorted(valid_problems):
            instances = sorted(valid_problems[problem_name], key=int)
            dim = int(metadata[problem_name]["n"])
            handle.write(
                f"{problem_name} ({problem_name}) [N {dim}] [M 1] "
                f"[PB_INSTANCE {' '.join(instances)}]\n"
            )

    with (sync_dir / "algo_selection.def").open("w", encoding="utf-8") as handle:
        for strategy in strategies:
            handle.write(
                f"{strategy} ({strategy}) [STATS_FILE_NAME stats.txt] "
                "[STATS_FILE_OUTPUT CNT_EVAL OBJ] [ADD_PBINSTANCE_TO_STATS_FILE yes]\n"
            )

    taus = {"1e-1": "0.1", "1e-3": "0.001", "1e-5": "0.00001"}
    with (sync_dir / "output_selection.def").open("w", encoding="utf-8") as handle:
        for label, value in taus.items():
            handle.write(
                f"DATA_PROFILE (DP_{label}) [y_select OBJ] [tau {value}] "
                f"[output_plain dp_{label}.txt] [output_latex dp_{label}.tex]\n"
            )
            handle.write(
                f"PERFORMANCE_PROFILE (PP_{label}) [y_select OBJ] [tau {value}] "
                f"[output_plain pp_{label}.txt] [output_latex pp_{label}.tex]\n"
            )


def command_format(args: argparse.Namespace) -> None:
    copied = format_results(args.source_dir.resolve(), args.target_dir.resolve(), args.strategies)
    print(f"formatted stats files: {copied}")


def command_sync(args: argparse.Namespace) -> None:
    problems, files = sync_results(
        args.source_dir.resolve(),
        args.sync_dir.resolve(),
        args.strategies,
        strict_complete=args.strict_complete,
    )
    print(f"synchronized problems: {problems}")
    print(f"synchronized stats files: {files}")


def command_runnerpost(args: argparse.Namespace) -> None:
    run_runnerpost(args.runnerpost_exe.resolve(), args.sync_dir.resolve())
    if args.style_tex:
        changed = style_tex_files(args.sync_dir.resolve(), remove_legends=not args.keep_legends)
        print(f"styled tex files: {changed}")
    if args.compile_tex:
        compile_tex(
            args.sync_dir.resolve(),
            args.pdf_dir,
            args.figure_title,
            template_dir=args.template_dir.resolve() if args.template_dir else None,
            main_plot_width=args.main_plot_width,
        )


def command_all(args: argparse.Namespace) -> None:
    if args.run_simulation:
        simulate(args)
    copied = format_results(args.results_dir.resolve(), args.formatted_dir.resolve(), args.strategies)
    print(f"formatted stats files: {copied}")
    problems, files = sync_results(
        args.formatted_dir.resolve(),
        args.sync_dir.resolve(),
        args.strategies,
        strict_complete=args.strict_complete,
    )
    print(f"synchronized problems: {problems}")
    print(f"synchronized stats files: {files}")
    if args.runnerpost_exe:
        command_runnerpost(args)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--strategies", nargs="+", default=DEFAULT_STRATEGY_ORDER, choices=list(STRATEGIES))


def add_sim_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--problems", type=parse_problem_list, default=DEFAULT_PROBLEMS)
    parser.add_argument(
        "--auto-select-constrained",
        action="store_true",
        help="Select available constrained CUTEst problems automatically.",
    )
    parser.add_argument("--auto-prefix", default="HS")
    parser.add_argument("--auto-count", type=positive_int, default=40)
    parser.add_argument("--min-dim", type=positive_int, default=1)
    parser.add_argument("--max-dim", type=positive_int, default=10)
    parser.add_argument("--n-subinstances", type=positive_int, default=2)
    parser.add_argument("--seed-stride", type=positive_int, default=123)
    parser.add_argument("--max-bb-eval", type=positive_int, default=500)
    parser.add_argument("--workers", type=positive_int, default=4)
    parser.add_argument("--mp-context", choices=["spawn", "forkserver", "fork"], default="spawn")
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--skip-unconstrained", action="store_true", default=True)
    parser.add_argument(
        "--start-mode",
        choices=["cutest", "feasible", "mixed"],
        default="feasible",
        help="cutest: repeat CUTEst x0; feasible: random feasible starts first; mixed: include CUTEst x0 plus random feasible starts.",
    )
    parser.add_argument("--start-seed", type=int, default=20240519)
    parser.add_argument("--candidate-starts", type=positive_int, default=2000)
    parser.add_argument("--feasibility-tol", type=float, default=1e-8)
    parser.add_argument("--start-radius", type=float, default=5.0)
    parser.add_argument("--clean", action="store_true")


def add_runnerpost_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--runnerpost-exe", type=Path, required=True)
    parser.add_argument("--sync-dir", type=Path, required=True)
    parser.add_argument("--style-tex", action="store_true")
    parser.add_argument("--keep-legends", action="store_true")
    parser.add_argument("--compile-tex", action="store_true")
    parser.add_argument("--pdf-dir", default="pdfs")
    parser.add_argument("--figure-title", default="CUTEst constrained ordering profiles")
    parser.add_argument("--template-dir", type=Path)
    parser.add_argument("--main-plot-width", default=r"0.32\textwidth")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    simulate_parser = subparsers.add_parser("simulate")
    add_sim_args(simulate_parser)
    simulate_parser.set_defaults(func=simulate)

    format_parser = subparsers.add_parser("format")
    format_parser.add_argument("--source-dir", type=Path, required=True)
    format_parser.add_argument("--target-dir", type=Path, required=True)
    add_common_args(format_parser)
    format_parser.set_defaults(func=command_format)

    sync_parser = subparsers.add_parser("sync")
    sync_parser.add_argument("--source-dir", type=Path, required=True)
    sync_parser.add_argument("--sync-dir", type=Path, required=True)
    sync_parser.add_argument("--strict-complete", action="store_true")
    add_common_args(sync_parser)
    sync_parser.set_defaults(func=command_sync)

    runnerpost_parser = subparsers.add_parser("runnerpost")
    add_runnerpost_args(runnerpost_parser)
    runnerpost_parser.set_defaults(func=command_runnerpost)

    all_parser = subparsers.add_parser("all")
    all_parser.add_argument("--run-simulation", action="store_true")
    all_parser.add_argument("--formatted-dir", type=Path, required=True)
    all_parser.add_argument("--strict-complete", action="store_true")
    add_sim_args(all_parser)
    add_runnerpost_args(all_parser)
    add_common_args(all_parser)
    all_parser.set_defaults(func=command_all)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except Exception:
        pass
    raise SystemExit(main())
