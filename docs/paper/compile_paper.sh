#!/bin/bash
dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )" # path to current file
# repodir=$(builtin cd dir; cd ..; pwd)
repodir=$(builtin cd dir; pwd)
# repodir=$(builtin cd $docsdir; pwd)
echo $repodir
docker run --rm \
    --volume $repodir:/data \
    --user $(id -u):$(id -g) \
    --env JOURNAL=joss \
    openjournals/paperdraft
mv $repodir/paper.pdf $dir/