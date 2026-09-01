import argparse
import json
import os

for variable in [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]:
    os.environ.setdefault(variable, "1")
os.environ.setdefault("MPLCONFIGDIR", str(__import__("pathlib").Path(__file__).resolve().parent / ".mplconfig"))

from modules.workflow import run_workflow


def main():
    parser = argparse.ArgumentParser(description="Run the locked South German Credit workflow")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--plot-root", default=None)
    parser.add_argument("--verification", action="store_true")
    args = parser.parse_args()
    result = run_workflow(args.output_root, args.plot_root, verification=args.verification)
    print(json.dumps({"code_version": result["code_version"], "output_root": result["output_root"]}, indent=2))


if __name__ == "__main__":
    main()
