#!/usr/bin/env bash

set -x
PARTITION=$1
JOB_NAME=$2
CONFIG=$3
GPUS=${GPUS:-$4}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
PORT=`expr $RANDOM % 40000 + 1000`
echo "port: $PORT"
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:5}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --exclude SH-IDC1-10-140-0-[164,169,171,173,207],SH-IDC1-10-140-1-[48,49,69,107,119] \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    --quotatype=reserved \
    ${SRUN_ARGS} \
    python -Xfaulthandler -u tools/train.py ${CONFIG}  --launcher="slurm" ${PY_ARGS}
