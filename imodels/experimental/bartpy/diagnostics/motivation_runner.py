import argparse
import subprocess
import os

from .motivation import DATASETS_REGRESSION, DATASETS_SYNTHETIC

PTH = "/accounts/campus/omer_ronen/projects/tree_shrink/imodels"
N_SAMPLES = 100


def parse_args():
    parser = argparse.ArgumentParser(description='BART Research motivation')
    parser.add_argument('script', metavar='script', type=str,
                        help='script to run')
    parser.add_argument('n_trees', metavar='n_trees', type=int,
                        help='number of trees')

    args = parser.parse_args()
    return args


def _get_config(dataset, analysis, script):
    config = f"--job-name=bart_{dataset}_{analysis} --mail-type=ALL --mail-user=omer_ronen@berkeley.edu " \
             f"-o {PTH}/scripts/logs/bart_{script}_{dataset}_{analysis}.out -e {PTH}/scripts/logs/bart_{script}_{dataset}_{analysis}.err" \
             f" -p jsteinhardt"
    return config


def _get_python(dataset, n_trees, n_samples, analysis, script):
    python_env = "/accounts/campus/omer_ronen/.conda/envs/imdls/bin/python"
    script = f"imodels.experimental.bartpy.diagnostics.{script}"
    cmd = f"{python_env} -m {script} {dataset} {n_trees} --n_samples {n_samples} --analysis {analysis}"
    return cmd


def main():
    args = parse_args()
    script = args.script
    for d in DATASETS_SYNTHETIC:
        cmd = f"sbatch {_get_config(d[0], 's', script)} {os.path.join(PTH, 'scripts', f'{script}.sh')} {d[0]} {args.n_trees} {N_SAMPLES} s"
        subprocess.run(cmd, shell=True)
    for d in DATASETS_REGRESSION:
        cmd = f"sbatch {_get_config(d[0], 'i', script)} {os.path.join(PTH, 'scripts', f'{script}.sh')} {d[0]} {args.n_trees} {N_SAMPLES} i"
        subprocess.run(cmd, shell=True)


if __name__ == "__main__":
    main()
