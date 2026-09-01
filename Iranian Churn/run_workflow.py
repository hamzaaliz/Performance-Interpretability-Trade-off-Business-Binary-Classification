import argparse
import json
import os
import warnings

for variable in [
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
]:
    os.environ[variable] = "1"
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(__import__("pathlib").Path(__file__).resolve().parent / ".mplconfig"),
)

for message in [
    "divide by zero encountered in matmul",
    "overflow encountered in matmul",
    "invalid value encountered in matmul",
]:
    warnings.filterwarnings("ignore", message=message, category=RuntimeWarning, module=r"sklearn\..*")

from modules.workflow import run_workflow


def main():
    parser = argparse.ArgumentParser(description="Run the locked Iranian Churn workflow")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--plot-root", default=None)
    args = parser.parse_args()
    result = run_workflow(args.output_root, args.plot_root)
    print(json.dumps({"code_version": result["code_version"], "output_root": result["output_root"]}, indent=2))


if __name__ == "__main__":
    main()
