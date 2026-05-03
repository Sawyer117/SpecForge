# SpecForge on Ascend NPU —— 安装指南（HF backend）

> 在 Ascend NPU（Atlas A2 / 910b / A3）上跑 SpecForge DFlash + HF target backend
> 的可复现安装步骤。版本号和 SpecForge commit 都钉死在同目录的
> `requirements-ascend.txt` 里。
>
> 任何一步失败，把原始报错贴出来——见文末
> [常见问题排查](#常见问题排查)，列了五种最常见的卡点和应对。

---

## 装完你能拿到什么

- SpecForge 钉到 upstream commit `d5fb617f735db85a680876327f8173de0cb57c15`
- DFlash + HF target-model backend（不装 sglang，不 build NPU kernel）
- yunchang `0.6.4` 装上但冬眠（DFlash 默认参数下不会调用它）
- 可以直接拉起 `scripts/train_dflash.py --target-model-backend hf --attention-backend sdpa`

**故意不包含**的东西：
- `sglang` —— 仅 `--target-model-backend sglang` 需要
- `flash-attn` —— NPU 上没 build
- `triton-ascend` / `BiSheng` / `sgl-kernel-npu` / `DeepEP` —— 仅 sglang backend
  的 NPU kernel 需要

---

## 前置条件

- 机器上已有 **CANN ≥ 8.5.0** 安装。记下安装路径，待会儿当 `CANN_HOME` 用。
- **conda** (Miniconda/Anaconda) 已经在 `$PATH` 里。
- 能访问华为云 PyPI 镜像（`mirrors.huaweicloud.com`）和 GitHub
  （`github.com`）。
- Python 3.11 由步骤 1 装到 conda 环境里，不需要系统级。

---

## 安装步骤

每一步独立，单步失败修完那一步重跑即可。下面引用的所有钉死版本号都在
`docs/ascend_npu/requirements-ascend.txt` 里——这个文件随步骤 2 clone
下来就有。

### 步骤 0 —— Source CANN

```bash
export CANN_HOME=/path/to/your/CANN/8.5.0.x       # ← 改成你机器的实际 CANN 路径
source "$CANN_HOME/ascend-toolkit/set_env.sh"
[ -f "$CANN_HOME/nnal/asdsip/set_env.sh" ] && source "$CANN_HOME/nnal/asdsip/set_env.sh"
[ -f "$CANN_HOME/nnal/atb/set_env.sh" ]    && source "$CANN_HOME/nnal/atb/set_env.sh"
```

### 步骤 1 —— 建 conda 环境（Python 3.11）

```bash
conda create -p ./conda/specforge_npu python=3.11 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ./conda/specforge_npu
```

### 步骤 2 —— 拉 fork 的 `docs/ascend-npu` 分支

这个分支带：
- SpecForge 源码已经在钉死的 upstream commit `d5fb617`
- `docs/ascend_npu/` 下有本指南和 `requirements-ascend.txt`

```bash
git clone -b docs/ascend-npu https://github.com/Sawyer117/SpecForge.git
cd SpecForge
```

### 步骤 3 —— 从华为云镜像装 torch / torchvision / torch_npu

```bash
pip install torch==2.9.0 torchvision==0.24.0 \
    -i https://mirrors.huaweicloud.com/repository/pypi/simple/ \
    --trusted-host mirrors.huaweicloud.com

pip install torch_npu==2.9.0 \
    -i https://mirrors.huaweicloud.com/repository/pypi/simple/ \
    --trusted-host mirrors.huaweicloud.com
```

> 如果镜像上没有 `torch_npu==2.9.0`，看
> [常见问题 #1](#1-torch_npu290-镜像上找不到)。

### 步骤 4 —— 装其余钉死版本的依赖

```bash
# 把 torch / torchvision 行从 requirements 里去掉（步骤 3 已经装过）
sed -i '/^torch==/d; /^torchvision==/d' docs/ascend_npu/requirements-ascend.txt

pip install -r docs/ascend_npu/requirements-ascend.txt
```

> **不要 revert** 这个 sed。装完之后这个文件改动只在本地——`git status`
> 看到它被改了不要紧，将来想恢复用 `git checkout -- docs/ascend_npu/requirements-ascend.txt`。

### 步骤 5 —— editable 安装 SpecForge，**绕过它的 pyproject 依赖解析**

```bash
pip install -e . --no-deps
```

`--no-deps` **必须有**。否则 pip 会按 SpecForge 自己的 `pyproject.toml` 去满足
`torch==2.9.1`、`sglang==0.5.9` 这两条，跟你刚刚步骤 3–4 装的会冲突。

### 步骤 6 —— 验证

```bash
python - <<'PY'
import torch, torch_npu
from yunchang.globals import PROCESS_GROUP, set_seq_parallel_pg, HAS_FLASH_ATTN, HAS_NPU
import transformers
import specforge

print("torch                    :", torch.__version__)
print("torch_npu                :", torch_npu.__version__)
print("transformers             :", transformers.__version__)
print("yunchang.HAS_NPU         :", HAS_NPU)
print("yunchang.HAS_FLASH_ATTN  :", HAS_FLASH_ATTN)
print("torch.npu.is_available() :", torch.npu.is_available())
print("torch.npu.device_count() :", torch.npu.device_count())
print("specforge import OK")
PY
```

### 期望输出

```
torch                    : 2.9.0
torch_npu                : 2.9.0          (或 2.9.0.postN，看镜像实际有什么)
transformers             : 4.57.1
yunchang.HAS_NPU         : True
yunchang.HAS_FLASH_ATTN  : False
torch.npu.is_available() : True
torch.npu.device_count() : 8              (你机器实际 NPU 数)
specforge import OK
```

六行都对的话，环境就绪，可以开始训练。

---

## 常见问题排查

### 1. `torch_npu==2.9.0` 镜像上找不到

先放宽版本约束试试：

```bash
pip install torch_npu \
    -i https://mirrors.huaweicloud.com/repository/pypi/simple/ \
    --trusted-host mirrors.huaweicloud.com
```

记下 pip 实际选了哪个版本（比如 `torch_npu-2.9.0.post1`），同步更新
`requirements-ascend.txt`。如果连这条也失败，说明这个 wheel 不在公开镜像上
——找你 CANN 那边要内部 release 通道的 URL，或者去
<https://www.hiascend.com/document/redirect/CannCommercialDeveloperResource>
直接拿 wheel。

### 2. `numpy<2.0` 跟其他包冲突

pip resolver 报 `numpy` 冲突的话，把宽松约束换成具体版本：

```bash
pip install numpy==1.26.4
```

然后再跑步骤 4，这时候 numpy 已经钉好了。

### 3. `setuptools<81` 冲突

同样思路，钉到具体版本：

```bash
pip install "setuptools==80.9.0"
```

之所以要 `setuptools<81`，是因为 `torch_npu.dynamo.torchair` 还在 import
`pkg_resources`，setuptools 81+ 把它删了。

### 4. yunchang 不小心把 flash-attn 拉进来

如果你装的是 `pip install yunchang==0.6.4`（没带 `[flash]` extra），不该发生。
万一发生了，明确不带 deps 装：

```bash
pip install --no-deps yunchang==0.6.4
pip install "torch>=2.3.0"   # 这是 yunchang 唯一真正的硬依赖
```

### 5. `from yunchang.globals import ...` 报 `ImportError`

把完整 traceback 抓出来。常见原因是 yunchang 的 `__init__.py` 跑
`from .ring import *`，传递触发了某个子模块需要一个本机不存在的包。报错原文给我
——yunchang 0.6.4 理论上 NPU 干净，这一步出问题非常值得追。

---

## 装完之后

1. 每次开新 shell，重新 source CANN（步骤 0 那一坨）。
2. 训练前激活 `transfer_to_npu` shim。shim 文件和封装好的启动脚本会在
   upstream-strategy 计划的 PR3 里加上；在那之前最小激活方式见
   `docs/ascend_npu/upstream_strategy_zh.md` §5.4。
3. 单机多 NPU 训练：用 `examples/run_qwen3_8b_dflash_online.sh`，加上
   `--target-model-backend hf --attention-backend sdpa`。多机扩展见
   `docs/ascend_npu/multi_node_training_zh.md`。
