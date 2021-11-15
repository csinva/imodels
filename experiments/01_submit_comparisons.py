from slurmpy import Slurm
from experiments.config.datasets import DATASETS_CLASSIFICATION, DATASETS_REGRESSION
DATASETS_ALL = DATASETS_CLASSIFICATION + DATASETS_REGRESSION

partition = 'high'
s = Slurm("compare_models", {"partition": partition})

for dset in DATASETS_ALL:
    param_str = 'source ~/chandan/imodels/imodels_env/bin/activate; '
    param_str += 'python3 ~/chandan/imodels/experiments/01_run_comparisons.py '
    param_str += f'--dataset {dset[0]} '
    param_str += f'--model sap '
#     print(param_str)
    s.run(param_str)