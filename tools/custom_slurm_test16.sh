#!/usr/bin/env bash

set -x
CKPT_PATH=/mnt/lustre/lianqing/data/mmdet3d-DfM/work_dirs
PARTITION=mm_det
JOB_NAME=$1
CONFIG=$2
TASK=$JOB_NAME
WORK_DIR=${CKPT_PATH}/${TASK}
GPUS=${GPUS:-$3}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
PORT=14555
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:4}
CKPT=${CKPT_PATH}/${TASK}/latest.pth

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    --quotatype=reserved \
    ${SRUN_ARGS} \
    python -Xfaulthandler -u tools/test.py ${CONFIG} --launcher="slurm" ${PY_ARGS}

