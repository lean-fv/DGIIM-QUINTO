"""
Generate error convergence graphs from CEC2017 results.

Iterates every algorithm subdirectory (results_*) under the given base
directory, then every dimension folder (dim_*) inside it. Reads all
milestone Excel files (results_N.xlsx) and produces one PNG per function,
showing the error convergence curve for each comparison algorithm.
Output PNGs are saved inside the corresponding dim_* folder.

Expected directory structure:
    results_cec17/
        results_BBOImproved/
            dim_10/
                results_1.xlsx
                results_2.xlsx
                ...
                results_100.xlsx
            dim_30/
                ...
            dim_50/
                ...

Requirements:
    pip install openpyxl matplotlib
    (both are listed in requirements.txt)

Usage:
    # From the virtual environment:
    python src/others/draw_errors.py [results_cec17_path] [--funcs F1,F2,...]

    # Generate graphs for ALL functions in ALL algorithms and dimensions:
    python src/others/draw_errors.py

    # Generate graphs only for F4 (Rosenbrock), F5 (Rastrigin), F21 (Composition1):
    python src/others/draw_errors.py --funcs 4,5,21

    # Specify a custom results directory and filter functions:
    python src/others/draw_errors.py /path/to/results_cec17 --funcs 4,5,21

Arguments:
    results_cec17_path  (optional) Path to the base results directory.
                        Defaults to ../../results_cec17 relative to this script.
    --funcs N1,N2,...   (optional) Comma-separated list of function numbers
                        to plot (e.g. 4,5,21). If omitted, all functions
                        found in the Excel files are plotted.

Output:
    For each (algorithm, dimension, function) combination, a PNG file is
    saved as: results_cec17/results_<alg>/dim_<D>/F<NN>_error.png
"""

from pathlib import Path
import sys

import openpyxl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    """
    @brief Parse command-line arguments.

    @return tuple (base_path, func_filter) where base_path is the results
            directory and func_filter is a set of function names to include
            (e.g. {'F04', 'F05', 'F21'}) or None for all functions.
    """
    base = None
    func_filter = None

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--funcs" and i + 1 < len(args):
            nums = [int(x.strip()) for x in args[i + 1].split(",")]
            func_filter = {f"F{n:02d}" for n in nums}
            i += 2
        else:
            base = Path(args[i])
            i += 1

    if base is None:
        base = Path(__file__).resolve().parent.parent.parent / "results_cec17"

    return base, func_filter


def read_milestone_file(filepath):
    """
    @brief Read one milestone Excel file and return its data.

    @param filepath Path to the results_N.xlsx file.
    @return tuple (algorithms, functions_data) where algorithms is a list
            of column header strings and functions_data is a dict mapping
            function name (e.g. 'F04') to a list of error values, one per algorithm.
    """
    wb = openpyxl.load_workbook(filepath, read_only=True)
    ws = wb.active

    rows = list(ws.iter_rows(values_only=True))
    wb.close()

    header_row = rows[0]
    algorithms = [str(h) for h in header_row[1:] if h is not None]

    functions_data = {}
    for row in rows[2:]:
        func_name = row[0]
        if func_name is None or not str(func_name).startswith("F"):
            continue
        values = list(row[1:1 + len(algorithms)])
        functions_data[str(func_name)] = values

    return algorithms, functions_data


def draw_function_graph(func_name, milestones, series, algorithms, dim_label, output_path):
    """
    @brief Draw and save one error convergence graph for a single function.

    @param func_name  Function identifier (e.g. 'F04').
    @param milestones Sorted list of milestone percentages (x-axis).
    @param series     Dict mapping algorithm name to list of error values (one per milestone).
    @param algorithms List of algorithm names for legend ordering.
    @param dim_label  Dimension string for the title (e.g. 'dim_30').
    @param output_path Path where the PNG will be saved.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for alg in algorithms:
        if alg in series:
            ax.plot(milestones, series[alg], marker="o", markersize=4, label=alg)

    ax.set_xlabel("Evaluations (%)")
    ax.set_ylabel("Error")
    ax.set_yscale("log")
    ax.set_title(f"{func_name} — {dim_label}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def process_dim_folder(dim_path, func_filter):
    """
    @brief Process one dimension folder: read all milestones and generate graphs.

    @param dim_path     Path object pointing to a dim_* directory containing results_N.xlsx files.
    @param func_filter  Set of function names to include (e.g. {'F04'}), or None for all.
    """
    milestone_files = sorted(dim_path.glob("results_*.xlsx"), key=lambda p: _milestone_key(p))

    if not milestone_files:
        return

    milestones = []
    all_data = {}
    algorithms = []

    for mf in milestone_files:
        milestone_num = _milestone_key(mf)
        milestones.append(milestone_num)

        algs, funcs_data = read_milestone_file(mf)
        if not algorithms:
            algorithms = algs

        for func_name, values in funcs_data.items():
            if func_filter is not None and func_name not in func_filter:
                continue
            if func_name not in all_data:
                all_data[func_name] = {alg: [] for alg in algorithms}
            for idx, alg in enumerate(algorithms):
                val = values[idx] if idx < len(values) else None
                all_data[func_name][alg].append(val)

    dim_label = dim_path.name
    generated = 0

    for func_name in sorted(all_data.keys(), key=lambda f: int(f[1:])):
        series = all_data[func_name]

        has_valid = False
        for alg in algorithms:
            if any(v is not None and v > 0 for v in series[alg]):
                has_valid = True

        if not has_valid:
            continue

        clean_series = {}
        for alg in algorithms:
            vals = series[alg]
            clean_vals = [v if (v is not None and v > 0) else 1e-14 for v in vals]
            clean_series[alg] = clean_vals

        output_path = dim_path / f"{func_name}_error.png"
        draw_function_graph(func_name, milestones, clean_series, algorithms, dim_label, output_path)
        generated += 1

    print(f"    {dim_label}: {generated} graphs generated")


def _milestone_key(path):
    """
    @brief Extract the numeric milestone from a filename like results_30.xlsx.

    @param path Path object.
    @return int Milestone percentage.
    """
    stem = path.stem
    num_str = stem.replace("results_", "")
    return int(num_str)


def main():
    base, func_filter = parse_args()

    if not base.is_dir():
        print(f"Error: directory '{base}' does not exist", file=sys.stderr)
        sys.exit(1)

    if func_filter:
        print(f"Filtering functions: {sorted(func_filter)}")

    alg_dirs = sorted([d for d in base.iterdir() if d.is_dir() and d.name.startswith("results_")])

    if not alg_dirs:
        print(f"No algorithm directories found in '{base}'", file=sys.stderr)
        sys.exit(1)

    for alg_dir in alg_dirs:
        alg_name = alg_dir.name.replace("results_", "")
        print(f"Algorithm: {alg_name}")

        dim_dirs = sorted([d for d in alg_dir.iterdir() if d.is_dir() and d.name.startswith("dim_")])

        for dim_dir in dim_dirs:
            process_dim_folder(dim_dir, func_filter)

    print("\nDone.")


if __name__ == "__main__":
    main()
