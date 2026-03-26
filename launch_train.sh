#!/usr/bin/env bash
set -e

########################## User Parameters ##########################
CFG=${1:?need config file}
RESOURCE_NUM_GPU=$2
DISTRIBUTED_NODE_COUNT=$3

########################## Compute Resource Parameters ########################
GPU_PER_NODE=${RESOURCE_NUM_GPU:?unset}
NODE_COUNT=${DISTRIBUTED_NODE_COUNT:?unset}
NODE_RANK=${DISTRIBUTED_NODE_RANK:-0}          # rank0 defaults to 0
MASTER_FQDN=${DISTRIBUTED_MASTER_HOSTS:?unset} # injected by Luban
MASTER_PORT=${LUBAN_AVAILBLE_PORT_0:-29500}

########################## Resolve Master Node IPv4 ###################
# Take the first FQDN segment before ',' -> resolve to IP; exit immediately on failure
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
export GLOO_DISABLE_IPV6=1     # disable IPv6 completely
export NCCL_DEBUG=WARN
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $(($GPU_PER_NODE - 1)))


conda activate colavla

########################## Print Parameters ##########################
echo "========== Distributed Launch =========="
echo "CFG            : $CFG"
# echo "WORKDIR        : $WORKDIR"
echo "GPUS / node    : $GPU_PER_NODE"
echo "NNODES         : $NODE_COUNT"
echo "NODE_RANK      : $NODE_RANK"
echo "MASTER_ADDR    : $MASTER_ADDR"
echo "MASTER_PORT    : $MASTER_PORT"
echo "NET_IF         : $NET_IF"
echo "========================================"

########################## Torchrun Launch #####################
export DISTRIBUTED_NODE_RANK=$NODE_RANK
export DISTRIBUTED_NODE_COUNT=$NODE_COUNT

PYTHONPATH="$(pwd)":$PYTHONPATH \
torchrun \
  --nnodes=$NODE_COUNT \
  --node_rank=$NODE_RANK \
  --nproc_per_node=$GPU_PER_NODE \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  tools/train.py \
  $CFG \
  --seed 0 \
  --launcher pytorch
