#!/bin/bash
# ============================================================================
# SpecForge DFlash multi-node HSDP launcher — Qwen3-8B, HF backend, NPU.
#
# Differs from run_qwen3_8b_dflash_npu_multinode.sh by passing --use-hsdp,
# which switches the FSDP wrap from SHARD_GRAD_OP-over-WORLD to
# _HYBRID_SHARD_ZERO2 with a 2D (replicate=NNODES, shard=NPUS_PER_NODE) mesh:
#
#   NNODES=2, NPUS_PER_NODE=8  =>  shard intra-node (8-way), replicate inter-node (2-way)
#                                   = "FSDP8 within node + DDP2 across nodes"
#
# Cross-node traffic per iteration drops from ~2 * param_size (FSDP-WORLD)
# to ~2 * param_size / NPUS_PER_NODE (HSDP), i.e. ~8x less for 8-NPU nodes.
#
# Other than --use-hsdp, every env var and behaviour matches
# run_qwen3_8b_dflash_npu_multinode.sh.
#
# Usage:
#
#   # On node 0:
#   MASTER_ADDR=<node0_ip> NNODES=2 NODE_RANK=0 \
#       HCCL_SOCKET_IFNAME=enp67s0f0 \
#       bash docs/ascend_npu/run_qwen3_8b_dflash_npu_multinode_hsdp.sh
#
#   # On node 1:
#   MASTER_ADDR=<node0_ip> NNODES=2 NODE_RANK=1 \
#       HCCL_SOCKET_IFNAME=enp67s0f0 \
#       bash docs/ascend_npu/run_qwen3_8b_dflash_npu_multinode_hsdp.sh
# ============================================================================

set -euo pipefail

: "${MASTER_ADDR:?MASTER_ADDR is required: IP of node 0}"
: "${NNODES:?NNODES is required: total number of nodes}"
: "${NODE_RANK:?NODE_RANK is required: this nodes rank, 0..NNODES-1}"
: "${HCCL_SOCKET_IFNAME:?HCCL_SOCKET_IFNAME is required: run ip a to find a NIC}"

# ---- Paths ----
TARGET_MODEL=${TARGET_MODEL:-/share/canada_group_folder/ckpt/Qwen3-8B}
TRAIN_DATA=${TRAIN_DATA:-/share/canada_group_folder/dataset/perfectblend_train_regen.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-./outputs/qwen3-8b-dflash-npu-hsdp}

# ---- Devices ----
NPUS_PER_NODE=${NPUS_PER_NODE:-8}
ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

# ---- Torchrun rendezvous ----
MASTER_PORT=${MASTER_PORT:-29533}

# ---- Hyperparams ----
BATCH_SIZE=${BATCH_SIZE:-1}
ACCUMULATION_STEPS=${ACCUMULATION_STEPS:-1}
MAX_LENGTH=${MAX_LENGTH:-3072}
NUM_EPOCHS=${NUM_EPOCHS:-6}
LR=${LR:-6e-4}
NUM_ANCHORS=${NUM_ANCHORS:-512}
BLOCK_SIZE=${BLOCK_SIZE:-16}
LOSS_DECAY_GAMMA=${LOSS_DECAY_GAMMA:-7.0}
WARMUP_RATIO=${WARMUP_RATIO:-0.04}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}

# ---- Logging / checkpointing ----
LOG_INTERVAL=${LOG_INTERVAL:-50}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
REPORT_TO=${REPORT_TO:-tensorboard}

# ---- NPU runtime env ----
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-2}
export ACLNN_CACHE_LIMIT=${ACLNN_CACHE_LIMIT:-100000}
export NPU_ASD_ENABLE=${NPU_ASD_ENABLE:-0}
export ASCEND_LAUNCH_BLOCKING=${ASCEND_LAUNCH_BLOCKING:-0}

# ---- HCCL inter-node ----
export HCCL_SOCKET_IFNAME
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-$HCCL_SOCKET_IFNAME}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-3600}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-1800}

# ---- Repo root ----
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
ROOT_DIR=$(dirname "$(dirname "$SCRIPT_DIR")")
DRAFT_CONFIG=${DRAFT_CONFIG:-$ROOT_DIR/configs/qwen3-8b-dflash.json}

WORLD_SIZE=$((NPUS_PER_NODE * NNODES))

cat <<EOF
========== SpecForge DFlash multi-node HSDP training (NPU) ==========
ROOT_DIR                  : $ROOT_DIR
TARGET_MODEL              : $TARGET_MODEL
DRAFT_CONFIG              : $DRAFT_CONFIG
TRAIN_DATA                : $TRAIN_DATA
OUTPUT_DIR                : $OUTPUT_DIR
---- distributed ----
NNODES (replicate dim)    : $NNODES
NPUS_PER_NODE (shard dim) : $NPUS_PER_NODE
NODE_RANK                 : $NODE_RANK
WORLD_SIZE                : $WORLD_SIZE
MASTER_ADDR : MASTER_PORT : $MASTER_ADDR : $MASTER_PORT
ASCEND_RT_VISIBLE_DEVICES : $ASCEND_RT_VISIBLE_DEVICES
HCCL_SOCKET_IFNAME        : $HCCL_SOCKET_IFNAME
---- training ----
BATCH_SIZE                : $BATCH_SIZE
MAX_LENGTH                : $MAX_LENGTH
NUM_EPOCHS                : $NUM_EPOCHS
LEARNING_RATE             : $LR
HSDP                      : ENABLED  (--use-hsdp)
=====================================================================
EOF

[[ -f "$DRAFT_CONFIG" ]] || { echo "ERROR: DRAFT_CONFIG not found: $DRAFT_CONFIG" >&2; exit 1; }
[[ -d "$TARGET_MODEL" ]] || { echo "ERROR: TARGET_MODEL dir not found: $TARGET_MODEL" >&2; exit 1; }
[[ -f "$TRAIN_DATA" ]]   || { echo "ERROR: TRAIN_DATA not found: $TRAIN_DATA" >&2; exit 1; }
ip a show "$HCCL_SOCKET_IFNAME" &> /dev/null || {
    echo "ERROR: NIC $HCCL_SOCKET_IFNAME not found on this host." >&2; exit 1;
}

mkdir -p "$OUTPUT_DIR"

cd "$ROOT_DIR"

ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES \
torchrun \
    --nproc_per_node "$NPUS_PER_NODE" \
    --nnodes "$NNODES" \
    --node_rank "$NODE_RANK" \
    --master_addr "$MASTER_ADDR" \
    --master_port "$MASTER_PORT" \
    scripts/train_dflash.py \
    --target-model-path "$TARGET_MODEL" \
    --draft-config-path "$DRAFT_CONFIG" \
    --train-data-path "$TRAIN_DATA" \
    --output-dir "$OUTPUT_DIR" \
    --target-model-backend hf \
    --attention-backend sdpa \
    --use-hsdp \
    --num-epochs "$NUM_EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --accumulation-steps "$ACCUMULATION_STEPS" \
    --max-length "$MAX_LENGTH" \
    --learning-rate "$LR" \
    --num-anchors "$NUM_ANCHORS" \
    --block-size "$BLOCK_SIZE" \
    --loss-decay-gamma "$LOSS_DECAY_GAMMA" \
    --warmup-ratio "$WARMUP_RATIO" \
    --max-grad-norm "$MAX_GRAD_NORM" \
    --chat-template qwen \
    --report-to "$REPORT_TO" \
    --log-interval "$LOG_INTERVAL" \
    --save-interval "$SAVE_INTERVAL" \
    --dist-timeout 60
