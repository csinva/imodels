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
    'grl', 
    'oner', 
    'brs'
]

if not args.test_only:
    
    for model in models:
        s.run(f'python experiments/compare_models.py --model {model} --cv')

    brl_job_ids = []
    for i in range(26):
        job_id = s.run(f'python experiments/compare_models.py --model brl --parallel_id {i} --cv')
        brl_job_ids.append(job_id)
    s.run(f'python experiments/combine.py --model brl', depends_on=brl_job_ids)

if not args.val_only:

    models += [
        'stbl_l2_mm0', 
        'stbl_l2_mm1',
        'stbl_l2_mm2',
        'stbl_l1_mm0', 
        'stbl_l1_mm1', 
        'stbl_l1_mm2'
        ]

    for model in models:
        s.run(f'python experiments/compare_models.py --model {model} --test')

    brl_job_ids = []
    for i in range(10):
        job_id = s.run(f'python experiments/compare_models.py --test --model brl --parallel_id {i}')
        brl_job_ids.append(job_id)
    s.run(f'python experiments/combine.py --test --model brl', depends_on=brl_job_ids)
