import subprocess
import os

from .motivation import DATASETS_REGRESSION

PTH = "/accounts/campus/omer_ronen/projects/tree_shrink/imodels"

if __name__ == "__main__":
    for d in DATASETS_REGRESSION:
        cmd = f"sbatch {os.path.join(PTH, 'scripts', 'motivation.sh')} {d[0]}"
        subprocess.run(cmd, shell=True)
