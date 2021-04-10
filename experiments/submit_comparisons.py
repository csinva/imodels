from slurmpy import Slurm
import argparse

partition = 'high'

s = Slurm("compare_models", {"partition": partition})

parser = argparse.ArgumentParser()
parser.add_argument('--val_only', action='store_true')
parser.add_argument('--test_only', action='store_true')
args = parser.parse_args()

models = [
    'random_forest', 
    'gradient_boosting', 
    'skope_rules', 
    'rulefit', 
    'fplasso', 
    'fpskope',
]

if not args.test_only:
    
    for model in models:
        s.run(f'python experiments/compare_models.py --model {model}')

    brl_job_ids = []
    for i in range(26):
        job_id = s.run(f'python experiments/compare_models.py --model brl --parallel_id {i}')
        brl_job_ids.append(job_id)
    s.run(f'python experiments/compare_models.py --combine --model brl', depends_on=brl_job_ids)

if not args.val_only:
    models += ['grl', 'oner', 'brs']

    for model in models:
        s.run(f'python experiments/compare_models.py --model {model} --test')
    
    brl_job_ids = []
    for i in range(8):
        job_id = s.run(f'python experiments/compare_models.py --test --model brl --parallel_id {i}')
        brl_job_ids.append(job_id)
    s.run(f'python experiments/compare_models.py --combine --test --model brl', depends_on=brl_job_ids)
