import subprocess

from .clalit import DATASETS_CLASSIFICATION

PY = "/accounts/campus/omer_ronen/.conda/envs/imdls/bin/python"


def _get_config(d):
    j_name = f"--job-name clalit_{d}"
    mail = "--mail-type ALL --mail-user omer_ronen@berkeley.edu"
    logs = f"-o /accounts/campus/omer_ronen/projects/tree_shrink/imodels/imodels/experimental/clalit/{d}.out" \
           f" -e /accounts/campus/omer_ronen/projects/tree_shrink/imodels/imodels/experimental/clalit/{d}.err"
    priority = "-p high"
    return f"{j_name} {mail} {logs} {priority}"


def main():
    for d in DATASETS_CLASSIFICATION:
        cmd = f"sbatch {_get_config(d[0])} {PY} -m imodels.experimental.clalit.clalit {d[0]}"
        subprocess.run(cmd, shell=True)


if __name__ == '__main__':
    main()
