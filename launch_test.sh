#!/usr/bin/env bash
set -e

########################## User Parameters ##########################
CONFIG=$1
CHECKPOINT=$2
GPUS=${3:-1}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

NAME=$(basename "$CONFIG" .py)
echo "$NAME"

########################## Compute Resource Parameters ########################
GPUS=${RESOURCE_NUM_GPU:-$GPUS}
NNODES=${DISTRIBUTED_NODE_COUNT:-$NNODES}
NODE_RANK=${DISTRIBUTED_NODE_RANK:-$NODE_RANK}
MASTER_FQDN=${DISTRIBUTED_MASTER_HOSTS:-$MASTER_ADDR}
PORT=${LUBAN_AVAILBLE_PORT_0:-$PORT}

########################## Resolve Master Node IP ######################
MASTER_ADDR=$(getent hosts "${MASTER_FQDN%%,*}" | awk '{print $1}' | head -n1)
[[ -z "$MASTER_ADDR" ]] && { echo "[ERROR] cannot resolve $MASTER_FQDN"; exit 1; }

########################## Network Interface Selection ######################
NET_IF=$(ip -o -4 route show to default | awk '{print $5}' | head -n1)

########################## NCCL/GLOO Environment ####################
export NCCL_SOCKET_IFNAME=$NET_IF
export GLOO_SOCKET_IFNAME=$NET_IF
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=3600
export GLOO_DISABLE_IPV6=1
export NCCL_DEBUG=WARN
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $(($GPUS - 1)))


conda activate colavla

########################## Print Parameters ######################
echo "========== Distributed Launch =========="
echo "CONFIG         : $CONFIG"
echo "CHECKPOINT     : $CHECKPOINT"
echo "GPUS / node    : $GPUS"
echo "NNODES         : $NNODES"
echo "NODE_RANK      : $NODE_RANK"
echo "MASTER_ADDR    : $MASTER_ADDR"
echo "PORT           : $PORT"
echo "NET_IF         : $NET_IF"
echo "========================================"

########################## Torchrun Launch #####################
export DISTRIBUTED_NODE_RANK=$NODE_RANK
export DISTRIBUTED_NODE_COUNT=$NNODES

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$PORT \
    --nproc_per_node=$GPUS \
    --use_env \
    $(dirname "$0")/tools/test.py \
    $CONFIG \
    $CHECKPOINT \
    --launcher pytorch \
    ${@:4}

# # evaluation
# cd /nfs/dataset-ofs-voyager-research/pqh/OmniDrive/evaluation
# echo "[INFO] Evaluating results at: ../results_planning/$NAME"
# python eval_planning.py --pred_path ../results_planning/$NAME/

CFG_NAME=$(basename $CONFIG) # mask_eva_lane_det_vlm_image_soft_mask_0.3.py
CFG_NAME=${CFG_NAME%.py} # mask_eva_lane_det_vlm_image_soft_mask_0.3

python evaluation/eval_planning_pkl.py \
    --base_path=data/nuscenes \
    --pred_path=results_planning/${CFG_NAME}/vla_results \
    --eval_classification
