#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=`expr $RANDOM % 40000 + 1000`
echo "port: $PORT"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
