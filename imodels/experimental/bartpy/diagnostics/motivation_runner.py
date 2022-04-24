import argparse
import subprocess
import os

from .motivation import DATASETS_REGRESSION, DATASETS_SYNTHETIC

PTH = "/accounts/campus/omer_ronen/projects/tree_shrink/imodels"


def parse_args():
    parser = argparse.ArgumentParser(description='BART Research motivation')
    parser.add_argument('n_trees', metavar='n_trees', type=int,
                        help='dataset to run sim over')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    for d in DATASETS_SYNTHETIC:
        cmd = f"sbatch {os.path.join(PTH, 'scripts', 'motivation.sh')} {d[0]} {args.n_trees}"
        subprocess.run(cmd, shell=True)
