import argparse
import subprocess
import os

from .motivation import DATASETS_REGRESSION, DATASETS_SYNTHETIC

PTH = "/accounts/campus/omer_ronen/projects/tree_shrink/imodels"
N_SAMPLES = 6000


def parse_args():
    parser = argparse.ArgumentParser(description='BART Research motivation')
    parser.add_argument('n_trees', metavar='n_trees', type=int,
                        help='dataset to run sim over')

    args = parser.parse_args()
    return args


def _get_config(dataset, analysis):
    config = f"--job-name=bart_{dataset}_{analysis} --mail-type=ALL --mail-user=omer_ronen@berkeley.edu " \
             f"-o {PTH}/scripts/logs/bart_{dataset}_{analysis}.out -e {PTH}/scripts/logs/bart_{dataset}_{analysis}.err" \
             f" -p jsteinhardt -C mem1024g"
    return config


def _get_python(dataset, n_trees, n_samples, analysis):
    python_env = "/accounts/campus/omer_ronen/.conda/envs/imdls/bin/python"
    script = "imodels.experimental.bartpy.diagnostics.motivation"
    cmd = f"{python_env} -m {script} {dataset} {n_trees} --n_samples {n_samples} --analysis {analysis}"
    return cmd


if __name__ == "__main__":
    args = parse_args()
    for d in DATASETS_SYNTHETIC:
        # python_cmd = _get_python(d[0], args.n_trees, N_SAMPLES, "s")
        cmd = f"sbatch {_get_config(d[0], 's')} {os.path.join(PTH, 'scripts', 'motivation.sh')} {d[0]} {args.n_trees} {N_SAMPLES} s"
        # cmd = f"sbatch {_get_config(d[0], 's')} {python_cmd}"
        subprocess.run(cmd, shell=True)
    for d in DATASETS_REGRESSION:
        cmd = f"sbatch {_get_config(d[0], 'i')} {os.path.join(PTH, 'scripts', 'motivation.sh')} {d[0]} {args.n_trees} {N_SAMPLES} i"
        # python_cmd = _get_python(d[0], args.n_trees, N_SAMPLES, "i")
        # cmd = f"sbatch {_get_config(d[0], 'i')} {python_cmd}"
        subprocess.run(cmd, shell=True)
