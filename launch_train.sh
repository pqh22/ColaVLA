#!/usr/bin/env bash
set -e

########################## 用户参数 ##########################
CFG=${1:?need config file}
RESOURCE_NUM_GPU=$2
DISTRIBUTED_NODE_COUNT=$3

########################## Luban 变量 ########################
GPU_PER_NODE=${RESOURCE_NUM_GPU:?unset}
NODE_COUNT=${DISTRIBUTED_NODE_COUNT:?unset}
NODE_RANK=${DISTRIBUTED_NODE_RANK:-0}          # rank0 默认为 0
MASTER_FQDN=${DISTRIBUTED_MASTER_HOSTS:?unset} # 由 Luban 注入
MASTER_PORT=${LUBAN_AVAILBLE_PORT_0:-29500}

########################## 解析主节点 IPv4 ###################
# 取逗号前第一段 FQDN -> 转 IP；失败直接退出
MASTER_ADDR=$(getent hosts "${MASTER_FQDN%%,*}" | awk '{print $1}' | head -n1)
[[ -z "$MASTER_ADDR" ]] && { echo "[ERROR] cannot resolve $MASTER_FQDN"; exit 1; }

########################## 网络接口选择 ######################
NET_IF=$(ip -o -4 route show to default | awk '{print $5}' | head -n1)

########################## NCCL/GLOO 环境 ####################
export NCCL_SOCKET_IFNAME=$NET_IF
export GLOO_SOCKET_IFNAME=$NET_IF
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=3600
export GLOO_DISABLE_IPV6=1     # 彻底不碰 IPv6
export NCCL_DEBUG=WARN
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $(($GPU_PER_NODE - 1)))


conda activate colavla

########################## 打印参数 ##########################
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

########################## Torchrun 启动 #####################
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
