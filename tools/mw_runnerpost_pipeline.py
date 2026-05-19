#!/usr/bin/env python3
"""More-Wild / NOMAD / RunnerPost benchmarking pipeline.

This script consolidates the workflow used to compare NOMAD EVAL_QUEUE_SORT
strategies on the More-Wild instances:

1. Optional PyNomad simulation.
2. Formatting raw NOMAD stats into RunnerPost's directory layout.
3. Strict synchronization of initial values across algorithms.
4. Generation of RunnerPost .def files.
5. Optional RunnerPost execution and LaTeX styling.

The defaults are intentionally conservative: incomplete problem instances are
skipped, and inconsistent f(x0) values are treated as errors unless explicitly
allowed by the user.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable


MW_DIMS: dict[int, int] = {
    1: 9, 2: 9, 3: 7, 4: 7, 5: 7, 6: 7, 7: 2, 8: 2, 9: 3, 10: 3,
    11: 4, 12: 4, 13: 2, 14: 2, 15: 3, 16: 3, 17: 4, 18: 3, 19: 6, 20: 6,
    21: 9, 22: 9, 23: 12, 24: 12, 25: 3, 26: 2, 27: 4, 28: 4, 29: 6, 30: 7,
    31: 8, 32: 9, 33: 10, 34: 11, 35: 10, 36: 5, 37: 11, 38: 11, 39: 8, 40: 10,
    41: 11, 42: 12, 43: 5, 44: 6, 45: 8, 46: 5, 47: 5, 48: 8, 49: 10, 50: 12,
    51: 12, 52: 8, 53: 8,
}

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

RUNNERPOST_FORMATS: dict[str, str] = {
    "QUADRATIC": "[FORMAT color=blue,mark=square]",
    "RANDOM": "[FORMAT color=orange,mark=*]",
    "LEXICO": "[FORMAT color=magenta,mark=triangle]",
    "DIR_LAST_SUCCESS": "[FORMAT color=red,mark=diamond]",
    "OMNISCIENT": "[FORMAT color=green,mark=star]",
    "REVERSE_OMNI": "[FORMAT color=black,mark=x]",
}

TEX_STYLES_BY_Y_INDEX: dict[int, str] = {
    1: r"solid, mark=square, mark size=1.5pt, mark repeat=20, color=blue",
    2: r"solid, mark=*, mark size=1.5pt, mark repeat=20, color=orange",
    3: r"solid, mark=triangle, mark size=1.5pt, mark repeat=20, color=magenta",
    4: r"solid, mark=diamond, mark size=1.5pt, mark repeat=20, color=red",
    5: r"thick, solid, mark=star, mark size=2pt, mark repeat=20, color=green!70!black",
    6: r"dashed, mark=x, mark size=1.5pt, mark repeat=20, color=black",
}

PROFILE_PDF_STEMS = [
    "dp_1e-1",
    "dp_1e-3",
    "dp_1e-5",
    "pp_1e-1",
    "pp_1e-3",
    "pp_1e-5",
]


@dataclasses.dataclass(frozen=True)
class StatsPoint:
    eval_count: int
    objective: float


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def parse_int_list(spec: str) -> list[int]:
    """Parse comma-separated integers and inclusive ranges such as 1,3,8-10."""
    values: set[int] = set()
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            if end < start:
                raise argparse.ArgumentTypeError(f"invalid decreasing range: {part}")
            values.update(range(start, end + 1))
        else:
            values.add(int(part))
    if not values:
        raise argparse.ArgumentTypeError("empty integer list")
    return sorted(values)


def parse_stats_file(path: Path) -> list[StatsPoint]:
    """Read a NOMAD/RunnerPost stats file containing at least BBE and OBJ."""
    points: list[StatsPoint] = []
    previous_eval = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                eval_count = int(parts[0])
                objective = float(parts[1])
            except ValueError:
                continue
            if eval_count <= 0:
                raise ValueError(f"{path}:{line_number}: non-positive evaluation count")
            if eval_count < previous_eval:
                raise ValueError(f"{path}:{line_number}: evaluation counts are not monotone")
            if not math.isfinite(objective):
                raise ValueError(f"{path}:{line_number}: non-finite objective")
            previous_eval = eval_count
            points.append(StatsPoint(eval_count, objective))
    if not points:
        raise ValueError(f"{path}: no numeric stats rows found")
    return points


def write_stats_file(path: Path, points: Iterable[StatsPoint]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for point in points:
            handle.write(f"{point.eval_count} {point.objective:.17g}\n")


def discover_stats_file(run_dir: Path) -> Path | None:
    candidates = sorted(
        p for p in run_dir.glob("stats*.txt") if p.is_file() and p.stat().st_size > 0
    )
    if not candidates:
        return None
    if len(candidates) > 1:
        names = ", ".join(p.name for p in candidates)
        raise ValueError(f"{run_dir}: multiple stats files found ({names})")
    return candidates[0]


def format_results(source_dir: Path, target_dir: Path, strategies: list[str]) -> int:
    if not source_dir.exists():
        raise FileNotFoundError(f"source directory does not exist: {source_dir}")
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True)

    copied = 0
    for problem_run_dir in sorted(p for p in source_dir.iterdir() if p.is_dir()):
        match = re.fullmatch(r"MW_(\d+)_(\d+)", problem_run_dir.name)
        if not match:
            continue
        mw_id_s, sub_id = match.groups()
        problem_name = f"MW_{mw_id_s}"
        for strategy in strategies:
            strategy_dir = problem_run_dir / strategy
            if not strategy_dir.exists():
                continue
            stats_file = discover_stats_file(strategy_dir)
            if stats_file is None:
                continue
            points = parse_stats_file(stats_file)
            write_stats_file(
                target_dir / strategy / problem_name / f"stats.{sub_id}.txt",
                points,
            )
            copied += 1
    return copied


def almost_equal(a: float, b: float, rel_tol: float, abs_tol: float) -> bool:
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def common_initial_value(
    series_by_strategy: dict[str, list[StatsPoint]],
    strategies: list[str],
    *,
    rel_tol: float,
    abs_tol: float,
    allow_rewrite: bool,
    label: str,
) -> float:
    initial_values = {strategy: series_by_strategy[strategy][0].objective for strategy in strategies}
    reference = initial_values[strategies[0]]
    mismatches = {
        strategy: value
        for strategy, value in initial_values.items()
        if not almost_equal(reference, value, rel_tol, abs_tol)
    }
    if mismatches and not allow_rewrite:
        details = ", ".join(f"{strategy}={value:.17g}" for strategy, value in initial_values.items())
        raise ValueError(
            f"{label}: inconsistent f(x0) values ({details}). "
            "Fix the simulation seeds/noise first, or pass --allow-f0-rewrite explicitly."
        )
    return reference


def sync_results(
    source_dir: Path,
    sync_dir: Path,
    strategies: list[str],
    *,
    rel_tol: float,
    abs_tol: float,
    allow_f0_rewrite: bool,
    strict_complete: bool,
) -> tuple[int, int]:
    if sync_dir.exists():
        shutil.rmtree(sync_dir)
    sync_dir.mkdir(parents=True)

    available: dict[str, set[tuple[str, str]]] = {}
    for strategy in strategies:
        strategy_dir = source_dir / strategy
        if not strategy_dir.exists():
            raise FileNotFoundError(f"missing strategy directory: {strategy_dir}")
        entries: set[tuple[str, str]] = set()
        for problem_dir in sorted(p for p in strategy_dir.iterdir() if p.is_dir()):
            for stats_file in problem_dir.glob("stats.*.txt"):
                parts = stats_file.name.split(".")
                if len(parts) >= 3:
                    entries.add((problem_dir.name, parts[1]))
        available[strategy] = entries

    common_entries = set.intersection(*(available[strategy] for strategy in strategies))
    if not common_entries:
        raise ValueError("no common problem instances found across all selected strategies")
    if strict_complete:
        for strategy in strategies:
            missing = sorted(common_entries.symmetric_difference(available[strategy]))
            if missing:
                formatted = ", ".join(f"{pb}/stats.{inst}.txt" for pb, inst in missing[:20])
                raise ValueError(f"{strategy}: incomplete data set; examples: {formatted}")

    valid_problems: dict[str, list[str]] = {}
    synchronized_files = 0
    for problem_name, instance_id in sorted(common_entries, key=lambda item: (item[0], int(item[1]))):
        series_by_strategy: dict[str, list[StatsPoint]] = {}
        for strategy in strategies:
            path = source_dir / strategy / problem_name / f"stats.{instance_id}.txt"
            series_by_strategy[strategy] = parse_stats_file(path)

        f0 = common_initial_value(
            series_by_strategy,
            strategies,
            rel_tol=rel_tol,
            abs_tol=abs_tol,
            allow_rewrite=allow_f0_rewrite,
            label=f"{problem_name}/stats.{instance_id}.txt",
        )

        valid_problems.setdefault(problem_name, []).append(instance_id)
        for strategy in strategies:
            normalized = [StatsPoint(1, f0)]
            normalized.extend(point for point in series_by_strategy[strategy] if point.eval_count != 1)
            write_stats_file(
                sync_dir / strategy / problem_name / f"stats.{instance_id}.txt",
                normalized,
            )
            synchronized_files += 1

    write_runnerpost_defs(sync_dir, strategies, valid_problems)
    return len(valid_problems), synchronized_files


def mw_id_from_problem_name(problem_name: str) -> int:
    match = re.fullmatch(r"MW_(\d+)", problem_name)
    if not match:
        raise ValueError(f"unexpected problem name: {problem_name}")
    return int(match.group(1))


def write_runnerpost_defs(
    sync_dir: Path,
    strategies: list[str],
    valid_problems: dict[str, list[str]],
) -> None:
    with (sync_dir / "problem_selection.def").open("w", encoding="utf-8") as handle:
        for problem_name in sorted(valid_problems, key=mw_id_from_problem_name):
            instances = sorted(valid_problems[problem_name], key=int)
            dim = MW_DIMS[mw_id_from_problem_name(problem_name)]
            handle.write(
                f"{problem_name} ({problem_name}) [N {dim}] [M 1] "
                f"[PB_INSTANCE {' '.join(instances)}]\n"
            )

    with (sync_dir / "algo_selection.def").open("w", encoding="utf-8") as handle:
        for strategy in strategies:
            style = RUNNERPOST_FORMATS[strategy]
            handle.write(
                f"{strategy} ({strategy}) [STATS_FILE_NAME stats.txt] "
                "[STATS_FILE_OUTPUT CNT_EVAL OBJ] [ADD_PBINSTANCE_TO_STATS_FILE yes] "
                f"{style}\n"
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


def profile_tex_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("dp*.tex")) + sorted(directory.glob("pp*.tex"))


def style_tex_files(directory: Path, remove_legends: bool = True) -> int:
    changed = 0
    for tex_file in profile_tex_files(directory):
        text = tex_file.read_text(encoding="utf-8")
        original = text
        text = apply_runnerpost_tex_fixes(text, remove_legends=remove_legends)
        text = style_addplot_commands(text)
        text = make_profile_tex_portable(text)

        if text != original:
            tex_file.write_text(text, encoding="utf-8")
            changed += 1
    return changed


def apply_runnerpost_tex_fixes(text: str, *, remove_legends: bool) -> str:
    """Apply the same cleanup as the historical sed post-processing block."""
    text = text.replace("\u00a0", " ")
    text = text.replace(r"\_", "_")

    fixed_lines = []
    for line in text.splitlines(keepends=True):
        if "title" in line:
            line = line.replace("_", r"\_")
        fixed_lines.append(line)
    text = "".join(fixed_lines)

    if remove_legends:
        text = re.sub(r"^.*\\addlegendentry.*\n?", "", text, flags=re.MULTILINE)
        text = re.sub(r"^.*\\legend.*\n?", "", text, flags=re.MULTILINE)
    return text


def style_addplot_commands(text: str) -> str:
    """Style each RunnerPost addplot command according to its y index."""

    def replace_command(match: re.Match[str]) -> str:
        command = match.group(0)
        y_match = re.search(r"y\s*index\s*=\s*(\d+)", command)
        if y_match is None:
            return command
        style = TEX_STYLES_BY_Y_INDEX.get(int(y_match.group(1)))
        if style is None:
            return command
        return re.sub(
            r"\\addplot\s*\[[^\]]*\]\s*table",
            rf"\\addplot [{style}] table",
            command,
            count=1,
        )

    return re.sub(r"\\addplot\b.*?;", replace_command, text, flags=re.DOTALL)


def make_profile_tex_portable(text: str) -> str:
    r"""Make RunnerPost profile TeX files compile on minimal TeX installs.

    Some GERAD server images do not ship standalone.cls. RunnerPost may emit
    profile figures with \documentclass{standalone}; replacing it with article
    keeps the figure compilable without changing the pgfplots body.
    """
    text = re.sub(
        r"\\documentclass(?:\[[^\]]*\])?\{(?:standalone|article)\}",
        r"\\documentclass{article}",
        text,
        count=1,
    )
    text = re.sub(r"^\\usepackage(?:\[[^\]]*\])?\{geometry\}\n?", "", text, flags=re.MULTILINE)
    text = text.replace(
        r"\documentclass{article}",
        "\\documentclass{article}\n"
        "\\usepackage[paperwidth=11cm,paperheight=8cm,margin=0.15cm]{geometry}",
        1,
    )
    if r"\pagestyle{empty}" not in text:
        text = text.replace(r"\begin{document}", "\\pagestyle{empty}\n\\begin{document}", 1)
    if r"\definecolor{orange}" not in text:
        text = text.replace(
            r"\begin{document}",
            "\\definecolor{orange}{rgb}{1,0.5,0}\n\\begin{document}",
            1,
        )
    if "every axis/.append style" not in text:
        text = text.replace(
            r"\begin{document}",
            "\\pgfplotsset{every axis/.append style={width=9.8cm,height=6.6cm,scale only axis}}\n"
            "\\begin{document}",
            1,
        )
    text = re.sub(r"^.*\\standaloneconfig.*\n?", "", text, flags=re.MULTILINE)
    return text


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def run_runnerpost(runnerpost_exe: Path, sync_dir: Path) -> None:
    subprocess.run(
        [
            str(runnerpost_exe),
            "algo_selection.def",
            "problem_selection.def",
            "output_selection.def",
        ],
        cwd=sync_dir,
        check=True,
    )


def write_main_tex(sync_dir: Path, pdf_dir: str, title: str, plot_width: str) -> Path:
    main_tex = sync_dir / "main.tex"
    escaped_title = latex_escape(title)
    main_tex.write_text(
        rf"""\documentclass{{article}}
\usepackage[margin=1cm,a4paper,landscape]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{tikz}}
\usepackage{{pgfplots}}
\usepackage{{xcolor}}
\pgfplotsset{{compat=newest}}
\pagestyle{{empty}}
\definecolor{{orange}}{{rgb}}{{1,0.5,0}}

\begin{{document}}
\begin{{center}}
{{\Large \textbf{{{escaped_title}}}}}\\[2ex]
\begin{{tabular}}{{ccc}}
    \multicolumn{{3}}{{c}}{{\Large \textbf{{Data Profiles}}}} \\[2ex]
    \includegraphics[width={plot_width}]{{{pdf_dir}/dp_1e-1}} &
    \includegraphics[width={plot_width}]{{{pdf_dir}/dp_1e-3}} &
    \includegraphics[width={plot_width}]{{{pdf_dir}/dp_1e-5}} \\
    \rule{{0pt}}{{6ex}} & & \\[-2ex]
    \multicolumn{{3}}{{c}}{{
        \fbox{{
            \begin{{tabular}}{{cl@{{\hspace{{8mm}}}}cl@{{\hspace{{8mm}}}}cl}}
                \tikz[baseline=-0.5ex]{{\draw[blue, thick] plot[mark=square] coordinates {{(-0.3,0) (0.3,0)}};}} & QUADRATIC MODEL &
                \tikz[baseline=-0.5ex]{{\draw[orange, thick] plot[mark=*] coordinates {{(-0.3,0) (0.3,0)}};}} & RANDOM &
                \tikz[baseline=-0.5ex]{{\draw[green!70!black, ultra thick] plot[mark=star] coordinates {{(-0.3,0) (0.3,0)}};}} & \textbf{{OMNISCIENT}} \\
                \rule{{0pt}}{{4ex}}
                \tikz[baseline=-0.5ex]{{\draw[magenta, thick] plot[mark=triangle] coordinates {{(-0.3,0) (0.3,0)}};}} & LEXICOGRAPHICAL &
                \tikz[baseline=-0.5ex]{{\draw[red, thick] plot[mark=diamond] coordinates {{(-0.3,0) (0.3,0)}};}} & DIR LAST SUCCESS &
                \tikz[baseline=-0.5ex]{{\draw[black, thick, dashed] plot[mark=x] coordinates {{(-0.3,0) (0.3,0)}};}} & \textbf{{REVERSE OMNI}} \\
            \end{{tabular}}
        }}
    }} \\
    \rule{{0pt}}{{8ex}} & & \\[-2ex]
    \multicolumn{{3}}{{c}}{{\Large \textbf{{Performance Profiles}}}} \\[2ex]
    \includegraphics[width={plot_width}]{{{pdf_dir}/pp_1e-1}} &
    \includegraphics[width={plot_width}]{{{pdf_dir}/pp_1e-3}} &
    \includegraphics[width={plot_width}]{{{pdf_dir}/pp_1e-5}} \\
\end{{tabular}}
\end{{center}}
\end{{document}}
""",
        encoding="utf-8",
    )
    return main_tex


def copy_template_files(template_dir: Path, sync_dir: Path, pdf_dir: str, plot_width: str) -> bool:
    if not template_dir.exists():
        raise FileNotFoundError(f"template directory does not exist: {template_dir}")

    copied_main = False
    for filename in ("defs.tex", "main.tex"):
        source = template_dir / filename
        if source.exists():
            shutil.copy(source, sync_dir / filename)
            copied_main = copied_main or filename == "main.tex"

    if copied_main:
        main_path = sync_dir / "main.tex"
        text = main_path.read_text(encoding="utf-8")
        text = make_main_tex_portable(text)
        text = resize_main_tex_graphics(text, plot_width)
        text = text.replace("pdfs/", f"{pdf_dir}/")
        main_path.write_text(text, encoding="utf-8")
    return copied_main


def make_main_tex_portable(text: str) -> str:
    text = re.sub(
        r"\\documentclass(?:\[[^\]]*\])?\{standalone\}",
        r"\\documentclass{article}",
        text,
        count=1,
    )
    if r"\documentclass{article}" in text and r"\usepackage[margin=1cm,a4paper,landscape]{geometry}" not in text:
        text = text.replace(
            r"\documentclass{article}",
            "\\documentclass{article}\n\\usepackage[margin=1cm,a4paper,landscape]{geometry}",
            1,
        )
    if r"\pagestyle{empty}" not in text:
        text = text.replace(r"\begin{document}", "\\pagestyle{empty}\n\\begin{document}", 1)
    text = re.sub(r"^.*\\standaloneconfig.*\n?", "", text, flags=re.MULTILINE)
    return text


def resize_main_tex_graphics(text: str, plot_width: str) -> str:
    return re.sub(
        r"width\s*=\s*0\.\d+\\textwidth",
        lambda _: f"width={plot_width}",
        text,
    )


def compile_tex(
    sync_dir: Path,
    pdf_dir: str,
    figure_title: str,
    *,
    template_dir: Path | None,
    main_plot_width: str,
) -> None:
    style_tex_files(sync_dir, remove_legends=True)
    tex_files = profile_tex_files(sync_dir)
    for tex_file in tex_files:
        text = tex_file.read_text(encoding="utf-8")
        portable = make_profile_tex_portable(text)
        if portable != text:
            tex_file.write_text(portable, encoding="utf-8")
        subprocess.run(["pdflatex", "-interaction=nonstopmode", tex_file.name], cwd=sync_dir, check=True)
    output_dir = sync_dir / pdf_dir
    output_dir.mkdir(exist_ok=True)
    for pdf_file in sorted(sync_dir.glob("dp*.pdf")) + sorted(sync_dir.glob("pp*.pdf")):
        shutil.move(str(pdf_file), output_dir / pdf_file.name)

    copied_main = False
    if template_dir is not None:
        copied_main = copy_template_files(template_dir, sync_dir, pdf_dir, main_plot_width)
    if not copied_main:
        write_main_tex(sync_dir, pdf_dir, figure_title, main_plot_width)
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "main.tex"], cwd=sync_dir, check=True)


def evaluate_external_bb(
    x: list[float],
    *,
    bb_exe: Path,
    instance: int,
    variant: int,
    run_dir: Path,
    env: dict[str, str],
    prefix: str,
    counter: int,
    fail_value: float,
    keep_eval_files: bool,
    eval_tmp_dir: Path | None,
) -> float:
    work_dir = eval_tmp_dir or run_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f"{prefix}_{counter}_",
        suffix=".txt",
        dir=work_dir,
        delete=False,
    ) as handle:
        x_file = Path(handle.name)
        handle.write(" ".join(f"{value:.17g}" for value in x) + "\n")
    try:
        completed = subprocess.run(
            [str(bb_exe), str(x_file), str(instance), str(variant)],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        if completed.returncode != 0:
            return fail_value
        tokens = completed.stdout.split()
        if not tokens or tokens[-1].lower() == "error":
            return fail_value
        return float(tokens[-1])
    except Exception:
        return fail_value
    finally:
        if not keep_eval_files:
            try:
                x_file.unlink()
            except FileNotFoundError:
                pass


def make_callback(
    *,
    bb_exe: Path,
    instance: int,
    variant: int,
    dim: int,
    run_dir: Path,
    env: dict[str, str],
    prefix: str,
    fail_value: float,
    keep_eval_files: bool,
    eval_tmp_dir: Path | None,
    reverse: bool = False,
):
    counter = {"value": 0}

    def callback(opt_param):
        counter["value"] += 1
        x = [opt_param.get_coord(i) for i in range(dim)]
        value = evaluate_external_bb(
            x,
            bb_exe=bb_exe,
            instance=instance,
            variant=variant,
            run_dir=run_dir,
            env=env,
            prefix=prefix,
            counter=counter["value"],
            fail_value=fail_value,
            keep_eval_files=keep_eval_files,
            eval_tmp_dir=eval_tmp_dir,
        )
        if reverse:
            value = -value
        opt_param.setBBO(str(value).encode("UTF-8"))
        return 1

    return callback


def run_py_nomad_task(task: tuple) -> str:
    (
        instance,
        variant,
        sub_id,
        strategy,
        sort_type,
        dim,
        results_dir_s,
        seed,
        bb_exe_s,
        lib_path_s,
        max_bb_eval,
        fail_value,
        keep_eval_files,
        eval_tmp_dir_s,
    ) = task

    import numpy as np
    import PyNomad

    results_dir = Path(results_dir_s)
    bb_exe = Path(bb_exe_s)
    run_dir = results_dir / f"MW_{instance}_{sub_id}" / strategy
    run_dir.mkdir(parents=True, exist_ok=True)
    eval_tmp_dir = None
    if eval_tmp_dir_s:
        eval_tmp_dir = Path(eval_tmp_dir_s) / f"MW_{instance}_{sub_id}" / strategy

    random_state = np.random.RandomState(seed)
    x0 = [float(value) for value in random_state.uniform(-1.0, 1.0, size=dim)]
    (run_dir / "x0.txt").write_text(" ".join(f"{value:.17g}" for value in x0) + "\n", encoding="utf-8")

    env = os.environ.copy()
    if lib_path_s:
        env["LD_LIBRARY_PATH"] = f"{lib_path_s}:{env.get('LD_LIBRARY_PATH', '')}"

    target_points = max(10, 4 * dim)
    params = [
        "BB_OUTPUT_TYPE OBJ",
        "DIRECTION_TYPE ORTHO 2N",
        f"TRIAL_POINT_MAX_ADD_UP {target_points}",
        f"MAX_BB_EVAL {max_bb_eval}",
        f"SEED {seed}",
        f"EVAL_QUEUE_SORT {sort_type}",
        "STATS_FILE stats.txt BBE OBJ",
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

    bb_callback = make_callback(
        bb_exe=bb_exe,
        instance=instance,
        variant=variant,
        dim=dim,
        run_dir=run_dir,
        env=env,
        prefix="x_bb",
        fail_value=fail_value,
        keep_eval_files=keep_eval_files,
        eval_tmp_dir=eval_tmp_dir,
    )

    previous_cwd = Path.cwd()
    os.chdir(run_dir)
    try:
        if strategy in SURROGATE_STRATEGIES:
            surrogate_callback = make_callback(
                bb_exe=bb_exe,
                instance=instance,
                variant=variant,
                dim=dim,
                run_dir=run_dir,
                env=env,
                prefix="x_surr",
                fail_value=fail_value,
                keep_eval_files=keep_eval_files,
                eval_tmp_dir=eval_tmp_dir,
                reverse=(strategy == "REVERSE_OMNI"),
            )
            result = PyNomad.optimize(bb_callback, x0, [], [], params, surrogate_callback)
        else:
            result = PyNomad.optimize(bb_callback, x0, [], [], params)
    finally:
        os.chdir(previous_cwd)

    return (
        f"done MW_{instance}_{sub_id} {strategy}: "
        f"bb={result.get('nb_evals')} f={result.get('f_best')}"
    )


def simulate(args: argparse.Namespace) -> None:
    results_dir = args.results_dir.resolve()
    if results_dir.exists() and args.clean:
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for instance in args.instances:
        dim = MW_DIMS[instance]
        for sub_id in range(1, args.n_subinstances + 1):
            seed = sub_id * args.seed_stride
            for strategy, sort_type in STRATEGIES.items():
                tasks.append(
                    (
                        instance,
                        args.variant,
                        sub_id,
                        strategy,
                        sort_type,
                        dim,
                        str(results_dir),
                        seed,
                        str(args.bb_exe.resolve()),
                        str(args.lib_path.resolve()) if args.lib_path else "",
                        args.max_bb_eval,
                        args.fail_value,
                        args.keep_eval_files,
                        str(args.eval_tmp_dir.resolve()) if args.eval_tmp_dir else "",
                    )
                )

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_py_nomad_task, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            print(future.result(), flush=True)


def command_format(args: argparse.Namespace) -> None:
    count = format_results(args.source_dir, args.target_dir, args.strategies)
    print(f"formatted stats files: {count}")


def command_sync(args: argparse.Namespace) -> None:
    n_problems, n_files = sync_results(
        args.source_dir,
        args.sync_dir,
        args.strategies,
        rel_tol=args.f0_rel_tol,
        abs_tol=args.f0_abs_tol,
        allow_f0_rewrite=args.allow_f0_rewrite,
        strict_complete=args.strict_complete,
    )
    print(f"synchronized problems: {n_problems}")
    print(f"synchronized stats files: {n_files}")


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
        if args.bb_exe is None:
            raise ValueError("--bb-exe is required when --run-simulation is set")
        simulate(args)
    formatted_dir = args.formatted_dir.resolve()
    sync_dir = args.sync_dir.resolve()
    count = format_results(args.results_dir.resolve(), formatted_dir, args.strategies)
    print(f"formatted stats files: {count}")
    n_problems, n_files = sync_results(
        formatted_dir,
        sync_dir,
        args.strategies,
        rel_tol=args.f0_rel_tol,
        abs_tol=args.f0_abs_tol,
        allow_f0_rewrite=args.allow_f0_rewrite,
        strict_complete=args.strict_complete,
    )
    print(f"synchronized problems: {n_problems}")
    print(f"synchronized stats files: {n_files}")
    if args.runnerpost_exe:
        run_runnerpost(args.runnerpost_exe.resolve(), sync_dir)
        if args.style_tex:
            changed = style_tex_files(sync_dir, remove_legends=not args.keep_legends)
            print(f"styled tex files: {changed}")
        if args.compile_tex:
            compile_tex(
                sync_dir,
                args.pdf_dir,
                args.figure_title,
                template_dir=args.template_dir.resolve() if args.template_dir else None,
                main_plot_width=args.main_plot_width,
            )


def add_common_strategy_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=DEFAULT_STRATEGY_ORDER,
        choices=list(STRATEGIES),
        help="Strategy names and plotting/order columns.",
    )


def add_sync_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--f0-rel-tol", type=float, default=1e-12)
    parser.add_argument("--f0-abs-tol", type=float, default=1e-12)
    parser.add_argument(
        "--allow-f0-rewrite",
        action="store_true",
        help="Rewrite f(x0) even when algorithms disagree. Use only after auditing why.",
    )
    parser.add_argument(
        "--strict-complete",
        action="store_true",
        help="Fail instead of silently keeping only complete problem-instance intersections.",
    )


def add_simulation_args(parser: argparse.ArgumentParser, *, require_bb_exe: bool = True) -> None:
    parser.add_argument("--bb-exe", type=Path, required=require_bb_exe)
    parser.add_argument("--lib-path", type=Path)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--variant", type=int, default=3, choices=[1, 2, 3, 4])
    parser.add_argument("--instances", type=parse_int_list, default=parse_int_list("1-53"))
    parser.add_argument("--n-subinstances", type=positive_int, default=2)
    parser.add_argument("--seed-stride", type=positive_int, default=123)
    parser.add_argument("--max-bb-eval", type=positive_int, default=500)
    parser.add_argument("--workers", type=positive_int, default=max(os.cpu_count() or 1, 1))
    parser.add_argument("--fail-value", type=float, default=1e20)
    parser.add_argument("--keep-eval-files", action="store_true")
    parser.add_argument(
        "--eval-tmp-dir",
        type=Path,
        help="Directory for transient blackbox input files; useful on quota-limited home filesystems.",
    )
    parser.add_argument("--clean", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    simulate_parser = subparsers.add_parser("simulate", help="Run PyNomad experiments.")
    add_simulation_args(simulate_parser, require_bb_exe=True)
    simulate_parser.set_defaults(func=simulate)

    format_parser = subparsers.add_parser("format", help="Format raw stats for RunnerPost.")
    format_parser.add_argument("--source-dir", type=Path, required=True)
    format_parser.add_argument("--target-dir", type=Path, required=True)
    add_common_strategy_arg(format_parser)
    format_parser.set_defaults(func=command_format)

    sync_parser = subparsers.add_parser("sync", help="Synchronize f(x0) and generate .def files.")
    sync_parser.add_argument("--source-dir", type=Path, required=True)
    sync_parser.add_argument("--sync-dir", type=Path, required=True)
    add_common_strategy_arg(sync_parser)
    add_sync_args(sync_parser)
    sync_parser.set_defaults(func=command_sync)

    runnerpost_parser = subparsers.add_parser("runnerpost", help="Run RunnerPost and style figures.")
    runnerpost_parser.add_argument("--runnerpost-exe", type=Path, required=True)
    runnerpost_parser.add_argument("--sync-dir", type=Path, required=True)
    runnerpost_parser.add_argument("--style-tex", action="store_true")
    runnerpost_parser.add_argument("--remove-legends", action="store_true")
    runnerpost_parser.add_argument("--keep-legends", action="store_true")
    runnerpost_parser.add_argument("--compile-tex", action="store_true")
    runnerpost_parser.add_argument("--pdf-dir", default="pdfs")
    runnerpost_parser.add_argument("--figure-title", default="More-Wild ordering profiles")
    runnerpost_parser.add_argument("--template-dir", type=Path)
    runnerpost_parser.add_argument("--main-plot-width", default=r"0.32\textwidth")
    runnerpost_parser.set_defaults(func=command_runnerpost)

    all_parser = subparsers.add_parser("all", help="Run the complete pipeline.")
    all_parser.add_argument("--run-simulation", action="store_true")
    all_parser.add_argument("--formatted-dir", type=Path, required=True)
    all_parser.add_argument("--sync-dir", type=Path, required=True)
    all_parser.add_argument("--runnerpost-exe", type=Path)
    all_parser.add_argument("--style-tex", action="store_true")
    all_parser.add_argument("--remove-legends", action="store_true")
    all_parser.add_argument("--keep-legends", action="store_true")
    all_parser.add_argument("--compile-tex", action="store_true")
    all_parser.add_argument("--pdf-dir", default="pdfs")
    all_parser.add_argument("--figure-title", default="More-Wild ordering profiles")
    all_parser.add_argument("--template-dir", type=Path)
    all_parser.add_argument("--main-plot-width", default=r"0.32\textwidth")
    add_common_strategy_arg(all_parser)
    add_sync_args(all_parser)
    add_simulation_args(all_parser, require_bb_exe=False)
    all_parser.set_defaults(func=command_all)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
