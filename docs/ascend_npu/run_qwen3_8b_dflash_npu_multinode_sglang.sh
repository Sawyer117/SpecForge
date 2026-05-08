#!/bin/bash
# ============================================================================
# SpecForge DFlash multi-node training launcher — Qwen3-8B, SGLANG backend, NPU.
#
# Combines the multi-node rendezvous + HCCL setup from
# run_qwen3_8b_dflash_npu_multinode.sh with the in-process sglang target
# backend from run_qwen3_8b_dflash_npu_sglang.sh.
#
# Mesh layout (defaults for 2 nodes × 8 NPUs = 16 ranks, TP=2 → DP=8):
#
#                 ┌──────────────── node 0 ────────────────┐  ┌──────────── node 1 ────────────┐
#   global rank   0  1   2  3   4  5   6  7                  8  9   10 11   12 13   14 15
#   target TP     [TP=2][TP=2] [TP=2] [TP=2]                 [TP=2][TP=2]  [TP=2] [TP=2]
#                  rep0  rep1   rep2   rep3                   rep4  rep5    rep6   rep7
#   draft DP      ░░░░░░░░░░ data-parallel across all 16 ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
#
# 8 target replicas total (4 per node), each TP-sharded across 2 NPUs. All
# TP groups stay intra-node (TP_SIZE divides NPUS_PER_NODE). The draft is
# data-parallel across all 16 ranks. No cross-node TP all-reduce — that
# path is not validated for sglang on NPU and would push per-layer
# collectives onto the slow inter-node link.
#
# TP=2 is a sane default for an 8B target on 64GB Atlas — TP=8 leaves only
# ~1B target params per NPU and burns most of the per-rank budget on a
# replica count of 2. TP=2 / DP=8 trades ~4x more target weight per NPU
# (~8GB vs ~2GB) for 4x more DP replicas, which usually wins for training.
# Override with TP_SIZE=4 / TP_SIZE=8 if you'd rather have headroom for KV.
#
# USAGE
#   On EVERY node, run this script with NODE_RANK set per node.
#
# Example (2 nodes, 8 NPUs each):
#
#   # On node 0:
#   MASTER_ADDR=172.27.2.112 NNODES=2 NODE_RANK=0 \
#       HCCL_SOCKET_IFNAME=enp67s0f0 \
#       bash docs/ascend_npu/run_qwen3_8b_dflash_npu_multinode_sglang.sh
#
#   # On node 1:
#   MASTER_ADDR=172.27.2.112 NNODES=2 NODE_RANK=1 \
#       HCCL_SOCKET_IFNAME=enp67s0f0 \
#       bash docs/ascend_npu/run_qwen3_8b_dflash_npu_multinode_sglang.sh
#
#   # Same, but also enable HSDP for the draft (intra-node shard,
#   # inter-node replicate — cuts cross-node grad traffic ~LOCAL_WORLD_SIZE x):
#   USE_HSDP=1 MASTER_ADDR=172.27.2.112 NNODES=2 NODE_RANK=0 \
#       HCCL_SOCKET_IFNAME=enp67s0f0 \
#       bash docs/ascend_npu/run_qwen3_8b_dflash_npu_multinode_sglang.sh
#
# REQUIRED env vars (no sane defaults — must set per environment):
#   MASTER_ADDR              IP of node 0 (must be reachable from all nodes)
#   NNODES                   number of nodes (e.g. 2, 4, 8)
#   NODE_RANK                this node's rank (0 .. NNODES-1)
#   HCCL_SOCKET_IFNAME       NIC name HCCL uses for inter-node traffic
#                            (run `ip a` on each node, pick the routable NIC,
#                             must be the same name on all nodes)
#
# Pre-flight (do once before first multi-node run):
#   - Single-node sglang launcher already passes end-to-end on each node
#     (validates the sgl_kernel_npu / triton-ascend / sglang stack).
#   - Single-node multinode HF launcher already passes end-to-end across
#     these nodes (validates the rendezvous + HCCL NIC choice).
#   - All nodes have the same SpecForge clone at the same commit and the same
#     conda env / CANN version sourced.
#   - Dataset path is identical on all nodes (shared FS or pre-staged).
#   - npu-smi info on each node — confirm 8 NPUs free per node before launch.
# ============================================================================

set -euo pipefail

# ---- Required ----
: "${MASTER_ADDR:?MASTER_ADDR is required: IP of node 0}"
: "${NNODES:?NNODES is required: total number of nodes}"
: "${NODE_RANK:?NODE_RANK is required: this nodes rank, 0..NNODES-1}"
: "${HCCL_SOCKET_IFNAME:?HCCL_SOCKET_IFNAME is required: run ip a to find a NIC}"

# ---- Paths ----
TARGET_MODEL=${TARGET_MODEL:-/share/canada_group_folder/ckpt/Qwen3-8B}
TRAIN_DATA=${TRAIN_DATA:-/share/canada_group_folder/dataset/perfectblend_train_regen.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-./outputs/qwen3-8b-dflash-npu-multinode-sglang}

# ---- Devices ----
NPUS_PER_NODE=${NPUS_PER_NODE:-8}
ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

# Target TP. Defaults to 2 → 8 replicas (DP=8) on a 16-rank world. TP must
# divide NPUS_PER_NODE so TP groups stay intra-node. Cross-node TP is
# possible by setting TP_SIZE > NPUS_PER_NODE but is NOT validated on NPU
# and will be slow because every TP all-reduce hits the inter-node link.
TP_SIZE=${TP_SIZE:-2}

# ---- Torchrun rendezvous ----
MASTER_PORT=${MASTER_PORT:-29533}

# ---- Hyperparams (same as single-node sglang defaults) ----
# Co-locating sglang + FSDP draft + activations on a 64 GB Atlas needs
# BATCH_SIZE=1; raising it OOMs once sglang's KV pool is reserved on top of
# training memory. Use ACCUMULATION_STEPS to grow effective batch.
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

# ---- SGLang backend hyperparams ----
# See run_qwen3_8b_dflash_npu_sglang.sh for the rationale on each value;
# in multi-node the per-node memory budget is unchanged, so the same
# defaults apply.
SGLANG_ATTENTION_BACKEND=${SGLANG_ATTENTION_BACKEND:-triton}
SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-0.4}

# ---- HSDP (optional) ----
# When set to 1, pass --use-hsdp to shard the draft intra-node and replicate
# inter-node. Cuts cross-node grad/all-gather traffic ~NPUS_PER_NODE x.
# Independent of the target TP mesh — fine to combine with TP=NPUS_PER_NODE.
USE_HSDP=${USE_HSDP:-0}

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

# ---- HCCL inter-node env (required for multi-node) ----
export HCCL_SOCKET_IFNAME
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-$HCCL_SOCKET_IFNAME}
# First inter-node all-reduce / sglang weight broadcast / kernel cache build
# is much slower than intra-node; bump these so HCCL doesn't time out before
# init finishes.
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-3600}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-1800}

# ---- Auto-locate SpecForge root ----
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
ROOT_DIR=$(dirname "$(dirname "$SCRIPT_DIR")")
DRAFT_CONFIG=${DRAFT_CONFIG:-$ROOT_DIR/configs/qwen3-8b-dflash.json}

WORLD_SIZE=$((NPUS_PER_NODE * NNODES))

# Sanity: TP_SIZE must divide WORLD_SIZE evenly
if (( WORLD_SIZE % TP_SIZE != 0 )); then
    echo "ERROR: WORLD_SIZE ($WORLD_SIZE) must be divisible by TP_SIZE ($TP_SIZE)" >&2
    exit 1
fi

# Warn (don't block) on cross-node TP — supported by the mesh setup but not
# a path we've validated on NPU.
if (( TP_SIZE > NPUS_PER_NODE )); then
    echo "WARN: TP_SIZE=$TP_SIZE exceeds NPUS_PER_NODE=$NPUS_PER_NODE." >&2
    echo "      Target TP will straddle nodes; per-layer all-reduce will" >&2
    echo "      hit the inter-node link. Path is not validated on NPU." >&2
fi

HSDP_FLAG=""
if [[ "$USE_HSDP" == "1" ]]; then
    HSDP_FLAG="--use-hsdp"
fi

cat <<EOF
======== SpecForge DFlash multi-node training (NPU, SGLANG) ========
ROOT_DIR                       : $ROOT_DIR
TARGET_MODEL                   : $TARGET_MODEL
DRAFT_CONFIG                   : $DRAFT_CONFIG
TRAIN_DATA                     : $TRAIN_DATA
OUTPUT_DIR                     : $OUTPUT_DIR
---- distributed ----
NNODES                         : $NNODES
NODE_RANK                      : $NODE_RANK
NPUS_PER_NODE                  : $NPUS_PER_NODE
WORLD_SIZE                     : $WORLD_SIZE
TP_SIZE (target)               : $TP_SIZE  $( (( TP_SIZE <= NPUS_PER_NODE )) && echo "(intra-node)" || echo "(CROSS-NODE — slow)")
HSDP (draft)                   : $( [[ "$USE_HSDP" == "1" ]] && echo ENABLED || echo disabled )
MASTER_ADDR : MASTER_PORT      : $MASTER_ADDR : $MASTER_PORT
ASCEND_RT_VISIBLE_DEVICES      : $ASCEND_RT_VISIBLE_DEVICES
HCCL_SOCKET_IFNAME             : $HCCL_SOCKET_IFNAME
HCCL_CONNECT_TIMEOUT           : $HCCL_CONNECT_TIMEOUT
HCCL_EXEC_TIMEOUT              : $HCCL_EXEC_TIMEOUT
---- training ----
BATCH_SIZE / MAX_LENGTH        : $BATCH_SIZE / $MAX_LENGTH
ACCUMULATION_STEPS             : $ACCUMULATION_STEPS
NUM_EPOCHS                     : $NUM_EPOCHS
LEARNING_RATE                  : $LR
---- sglang ----
SGLANG_ATTENTION_BACKEND       : $SGLANG_ATTENTION_BACKEND
SGLANG_MEM_FRACTION_STATIC     : $SGLANG_MEM_FRACTION_STATIC
=====================================================================
EOF

[[ -f "$DRAFT_CONFIG" ]] || { echo "ERROR: DRAFT_CONFIG not found: $DRAFT_CONFIG" >&2; exit 1; }
[[ -d "$TARGET_MODEL" ]] || { echo "ERROR: TARGET_MODEL dir not found: $TARGET_MODEL" >&2; exit 1; }
[[ -f "$TRAIN_DATA" ]]   || { echo "ERROR: TRAIN_DATA not found: $TRAIN_DATA" >&2; exit 1; }

# Sanity: NIC exists on this node
if ! ip a show "$HCCL_SOCKET_IFNAME" &> /dev/null; then
    echo "ERROR: NIC $HCCL_SOCKET_IFNAME not found on this host. Run 'ip a' to see options." >&2
    exit 1
fi

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
    --target-model-backend sglang \
    --tp-size "$TP_SIZE" \
    --attention-backend sdpa \
    --sglang-attention-backend "$SGLANG_ATTENTION_BACKEND" \
    --sglang-mem-fraction-static "$SGLANG_MEM_FRACTION_STATIC" \
    $HSDP_FLAG \
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
    --trust-remote-code \
    --dist-timeout 60
