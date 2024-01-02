#!/usr/bin/env bash

set -x
CKPT_PATH=/mnt/lustre/lianqing/data/work_dirs/det3d_v1.0
PARTITION=robot
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
    -x SH-IDC1-10-140-1-[76,102,165-166] \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    --quotatype=reserved \
    ${SRUN_ARGS} \
    python -Xfaulthandler -u tools/train.py ${CONFIG} --work-dir=${WORK_DIR} --launcher="slurm" ${PY_ARGS}

