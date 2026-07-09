"""
Extract CEC2017 milestone data from .txt result files into Excel summaries.

Iterates every algorithm subdirectory (results_*) under the given base
directory, reads all .txt files inside, groups by dimension, and produces
one Excel file per dimension containing mean errors at each milestone
for every function.

Usage:
    python extract.py [results_cec17_path]

If no path is given, defaults to ../../results_cec17 (relative to this script).
"""

from pathlib import Path

import os
import sys

import numpy as np
import pandas as pd


def process_algorithm_dir(alg_path):
    """
    @brief Process one algorithm directory: read .txt files and output Excel summaries.

    @param alg_path Path object pointing to a results_* directory containing .txt files.
    """
    fnames = [f.name for f in alg_path.iterdir() if f.suffix == ".txt"]

    if not fnames:
        print(f"  No .txt files found, skipping.")
        return

    saved_cwd = os.getcwd()
    os.chdir(alg_path)

    globaldf = pd.DataFrame({"funcid": [], "dim": [], "milestone": [], "error": []})

    for fname in fnames:
        df = pd.read_csv(fname)
        globaldf = pd.concat([globaldf, df], ignore_index=True)

    def funformat(funid):
        return "F{:02d}".format(funid)

    dim_df = globaldf.groupby(["dim"])

    for (dim, df) in dim_df:
        if isinstance(dim, tuple):
            dim = dim[0]

        milestones = [int(mil) for mil in np.unique(df.milestone)]
        funs = [int(fun) for fun in np.unique(df.funcid)]
        col_template = {"milestone": [], "dimension": []}
        for i in funs:
            col_template[funformat(i)] = []
        output_df = pd.DataFrame(col_template)

        if len(milestones) != 14:
            print(f"  Warning: unexpected milestone count ({len(milestones)}) for dim {int(dim)}: {milestones}", file=sys.stderr)
            os.chdir(saved_cwd)
            return

        if len(funs) < 1:
            print(f"  Error: no function data found for dim {int(dim)}", file=sys.stderr)
            os.chdir(saved_cwd)
            return

        mean_df = df.groupby(["funcid", "milestone"]).mean().reset_index().sort_values(by=["milestone", "funcid"])

        if mean_df.shape[1] != 4:
            print(f"  Error: column missing for dim {int(dim)}", file=sys.stderr)
            os.chdir(saved_cwd)
            return

        if mean_df.shape[0] != len(funs) * len(milestones):
            print(f"  Error: missing data for dim {int(dim)}", file=sys.stderr)
            os.chdir(saved_cwd)
            return

        for milestone in milestones:
            mildf = mean_df[mean_df.milestone == milestone]
            row_dict = {"milestone": [milestone], "dimension": [dim]}

            for (funid, (_, row)) in zip(funs, mildf.iterrows()):
                assert row.funcid == funid
                funcol = funformat(funid)
                value = float(row.error)
                row_dict[funcol] = value

            output_df = pd.concat([output_df, pd.DataFrame(row_dict)])

        out_fname = "results_cec2017_{}.xlsx".format(int(dim))
        output_df.to_excel(out_fname)
        print(f"  {out_fname}")

    os.chdir(saved_cwd)


def main():
    if len(sys.argv) > 1:
        base = Path(sys.argv[1])
    else:
        base = Path(__file__).resolve().parent.parent.parent / "results_cec17"

    if not base.is_dir():
        print(f"Error: directory '{base}' does not exist", file=sys.stderr)
        sys.exit(1)

    alg_dirs = sorted([d for d in base.iterdir() if d.is_dir() and d.name.startswith("results_")])

    if not alg_dirs:
        print(f"No algorithm directories found in '{base}'", file=sys.stderr)
        sys.exit(1)

    for alg_dir in alg_dirs:
        alg_name = alg_dir.name.replace("results_", "")
        print(f"Algorithm: {alg_name}")
        process_algorithm_dir(alg_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
