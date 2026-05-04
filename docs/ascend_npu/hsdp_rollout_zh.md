# HSDP rollout — 把多机 FSDP 切成"节点内 shard + 节点间 replicate"

> 这份分支（`feat/hsdp-multinode`）是从 `docs/ascend-npu` 分出来的实验分支。
> 现在的 `docs/ascend-npu` 里多机走 `SHARD_GRAD_OP over WORLD`，跨节点流量大；
> 这份分支换成 `_HYBRID_SHARD_ZERO2 + 2D mesh`，**双机 8 NPU 节点跨节点字节减少
> ~8×**。

---

## 改了什么

1. **`scripts/train_dflash.py`**: 新增 `--use-hsdp` flag。开启后 FSDP 用 2D mesh
   (replicate=NNODES, shard=LOCAL_WORLD_SIZE) + `_HYBRID_SHARD_ZERO2`。不开启
   时**完全保持原行为**（SHARD_GRAD_OP over WORLD），所以单机训练的 baseline
   不变。

2. **`docs/ascend_npu/run_qwen3_8b_dflash_npu_multinode_hsdp.sh`**: 多机 HSDP
   启动器。跟 `..._multinode.sh` 几乎一样，只是 torchrun 命令多传了 `--use-hsdp`。

## 通信账（双机 8 NPU/节点 = 16 卡，draft ≈ 1B params bf16）

| 路径 | 跨节点字节/iter | 跨节点 collective |
|---|---|---|
| 当前 main: `SHARD_GRAD_OP` over WORLD | ~4 GB（AG + RS over 16 卡）| 2 次 |
| 本分支: `_HYBRID_SHARD_ZERO2` + 2D mesh | **~250 MB**（AR on 1/8 grad shard）| 1 次 |

## 跑法

跟之前 multinode 完全一样，只是脚本换成 `..._hsdp.sh`：

### 节点 0

```bash
cd /home/a00652497/2026/SpecForge
git fetch origin && git checkout feat/hsdp-multinode

TRAIN_DATA=/share/canada_group_folder/dataset/perfectblend_train_10ksubset.jsonl \
MASTER_ADDR=172.27.2.112 \
NNODES=2 \
NODE_RANK=0 \
HCCL_SOCKET_IFNAME=enp67s0f0 \
bash docs/ascend_npu/run_qwen3_8b_dflash_npu_multinode_hsdp.sh
```

### 节点 1

```bash
cd /home/a00652497/2026/SpecForge
git fetch origin && git checkout feat/hsdp-multinode

TRAIN_DATA=/share/canada_group_folder/dataset/perfectblend_train_10ksubset.jsonl \
MASTER_ADDR=172.27.2.112 \
NNODES=2 \
NODE_RANK=1 \
HCCL_SOCKET_IFNAME=enp67s0f0 \
bash docs/ascend_npu/run_qwen3_8b_dflash_npu_multinode_hsdp.sh
```

## 验证 HSDP 真的生效（看一行 log）

启动后节点 0 的 log 里会多出：

```
HSDP enabled: replicate=2 (inter-node), shard=8 (intra-node)
```

如果你看不到这一行，说明 `--use-hsdp` 没传进去（脚本可能改坏了）。

## 期望性能差异

跟之前的 SHARD_GRAD_OP-over-WORLD 双机对比：

- **第一个 step 差异不明显**：HCCL kernel cache build 是一次性开销
- **稳态后**：跨节点 AR 远小于 AG+RS，**吞吐应该提高 30%–80%**（具体看你节点间 NIC 速度）
- **训出来的模型质量等价**（甚至略好，因为通信少了 → 等同 batch 训得更稳）

## 验完之后做什么

1. 用 `compare_decoding.py` / `compare_draft_predictions.py` 对比这分支训的 ckpt
   跟 `docs/ascend-npu` 训的 ckpt：argmax match >70% 就是 PASS。
2. 跑通后这分支可以作为 default 多机路径，把 `_multinode.sh` 替换成 `_hsdp.sh`
   的引用，但**先不动 `docs/ascend-npu`**——保持那条 well-tested 主线。
3. **不要把这分支合到 docs/ascend-npu**，等实测稳定 + 几次回归通过再合。

## 单机想试 HSDP 也行

虽然单机时"replicate=1"，HSDP 退化成普通 FSDP，但可以验证 flag 不会爆：

```bash
USE_HSDP_SINGLE_NODE=1 bash docs/ascend_npu/run_qwen3_8b_dflash_npu.sh
```

（但这个单机脚本目前没接 `--use-hsdp` flag，要小改一下；先在多机上验证更直接。）
