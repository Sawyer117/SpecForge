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

### 步骤 3 —— 一次装齐所有 Python 依赖（含 torch / torch_npu）

```bash
pip install -r docs/ascend_npu/requirements-ascend.txt
```

`requirements-ascend.txt` 头部已经声明：

```
--index-url https://mirrors.huaweicloud.com/repository/pypi/simple/
--trusted-host mirrors.huaweicloud.com
```

所以 torch、torchvision、torch_npu 以及其他所有包**都从华为云镜像拉**——
镜像上既有 PyPI 完整副本，也有 Ascend 的 `torch_npu` wheel，**一条命令搞定**。

> 如果镜像上没有 `torch_npu==2.9.0`，看
> [常见问题 #1](#1-torch_npu290-镜像上找不到)。

### 步骤 4 —— 装 sglang（即使你只用 HF backend 也必须装）

SpecForge `specforge/modeling/target/eagle3_target_model.py:5` 有一句：

```python
import sglang.srt.managers.mm_utils as mm_utils
```

是顶层硬 import，`specforge/modeling/target/__init__.py` 又会顶层 import
`eagle3_target_model`。所以**只要你 `import specforge`，sglang 就必须能 import**
——不管你跑训练时传的是 `--target-model-backend hf` 还是 `sglang`。

我们不装 PyPI 上的 `sglang==0.5.9`（拉太多 GPU-tagged 依赖），也不 copy 同事 fork
里的 sglang（继承不该继承的 patch）。**正解是从 upstream `sgl-project/sglang`
取一个已验证 commit**——同事 vendored 那份的版本字符串是
`0.5.6.post3.dev2770+g4926ca275`，里面 `g4926ca275` 就是上游 commit 短哈希。

```bash
# 1. 离开 SpecForge 目录，clone upstream sglang
cd ..
git clone https://github.com/sgl-project/sglang.git
cd sglang

# 2. checkout 到验证过的 commit
git checkout 4926ca275
git log --oneline 4926ca275 -1     # 确认 commit 存在

# 3. 看一下 python/ 下有几份 pyproject 备选
ls python/pyproject*.toml
# 期望: pyproject.toml  pyproject_cpu.toml  pyproject_npu.toml  pyproject_xpu.toml
# 如果只看到 pyproject.toml 一份，跳到 [常见问题 #6](#6-sglang-装不上)。

# 4. 用 NPU 版的 pyproject（先备份默认那份方便回退）
cp python/pyproject.toml python/pyproject.toml.bak
cp python/pyproject_npu.toml python/pyproject.toml

# 5. editable 安装。[srt_npu] 这个 extra 是空的（pyproject_npu.toml 里 srt_npu = []），
#   写不写都一样，遵循 upstream 习惯写上：
pip install -e "python[srt_npu]" \
    -i https://mirrors.huaweicloud.com/repository/pypi/simple/ \
    --trusted-host mirrors.huaweicloud.com

# 6. 验证 sglang import 通了
python -c "
import sglang
print('sglang version:', sglang.__version__)
import sglang.srt.managers.mm_utils
print('mm_utils import OK')
"
# 期望版本类似: 0.5.6.dev<N>+g4926ca275  （没有 'post3' 后缀；那是同事的 build metadata）

# 7. 回 SpecForge 目录，准备步骤 5
cd ../SpecForge
```

> 装这一步**第一次不要加 `--no-deps`**——让 pip 把 sglang 的依赖装齐，看到底
> 哪些能装上。如果某条依赖（典型 `flashinfer-python` / `sgl-kernel` /
> `vllm-flash-attn`）卡死编译失败，看 [常见问题 #6](#6-sglang-装不上)。

### 步骤 5 —— editable 安装 SpecForge，**绕过它的 pyproject 依赖解析**

```bash
pip install -e . --no-deps
```

`--no-deps` **必须有**。否则 pip 会按 SpecForge 自己的 `pyproject.toml` 去满足
`torch==2.9.1`、`sglang==0.5.9` 这两条——前者跟你刚装的 2.9.0 冲突、后者跟你
刚从源码装的 commit 版冲突，两条都会被 pip 强行覆盖装。

### 步骤 6 —— 验证

```bash
python - <<'PY'
import torch, torch_npu
from yunchang.globals import PROCESS_GROUP, set_seq_parallel_pg, HAS_FLASH_ATTN, HAS_NPU
import transformers
import sglang
import specforge

print("torch                    :", torch.__version__)
print("torch_npu                :", torch_npu.__version__)
print("transformers             :", transformers.__version__)
print("sglang                   :", sglang.__version__)
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
sglang                   : 0.5.6.dev<N>+g4926ca275
yunchang.HAS_NPU         : True
yunchang.HAS_FLASH_ATTN  : False
torch.npu.is_available() : True
torch.npu.device_count() : 8              (你机器实际 NPU 数)
specforge import OK
```

各行都对的话，环境就绪，可以开始训练。

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

### 6. sglang 装不上

#### 6a. `git checkout 4926ca275` 报 `unknown revision`

upstream 的浅 clone 默认不带全历史。换深 clone：

```bash
git clone --no-single-branch https://github.com/sgl-project/sglang.git
# 或者已经 clone 完之后补：
git fetch --unshallow
```

#### 6b. `ls python/pyproject*.toml` 只看到 `pyproject.toml` 一份

说明这个 commit 太老，多平台 pyproject 还没引入。两条退路：

退路 1，找最早引入 `pyproject_npu.toml` 的 upstream commit：

```bash
git log --diff-filter=A --oneline -- python/pyproject_npu.toml
# 取最早那个 commit hash，重做 git checkout
```

退路 2，直接装 PyPI 上 SpecForge upstream 钉的版本：

```bash
pip install sglang==0.5.9 --no-deps \
    -i https://mirrors.huaweicloud.com/repository/pypi/simple/ \
    --trusted-host mirrors.huaweicloud.com

# 然后根据 import 报错逐个补缺包
python -c "import sglang.srt.managers.mm_utils"
```

#### 6c. `pip install -e python` 卡在某条 GPU-only 依赖编译失败

典型出问题的：`flashinfer-python`、`sgl-kernel`、`vllm-flash-attn`。这些在 aarch64
没有 wheel，会试图编译 CUDA C++，缺 nvcc 就死。**用 `--no-deps` 跳过依赖解析**：

```bash
pip install -e "python[srt_npu]" --no-deps \
    -i https://mirrors.huaweicloud.com/repository/pypi/simple/ \
    --trusted-host mirrors.huaweicloud.com

# 然后用 import 验证哪些 dep 是真的 import-time 必需的，逐个补：
python -c "import sglang.srt.managers.mm_utils"
# 若报 ModuleNotFoundError: 'X'，pip install X 即可
```

最常见会缺：`compressed-tensors` / `xgrammar` / `uvloop` / `uvicorn` / `fastapi`
/ `msgspec` / `partial_json_parser` / `outlines` / `interegular` / `llguidance`
/ `anthropic` / `prometheus-client` / `pyzmq` / `setproctitle` / `tiktoken` /
`timm` / `smg-grpc-proto` / `hf_transfer` / `av` / `decord2` / `soundfile` /
`grpcio`。这些大多有 aarch64 wheel，能装。`flashinfer-python` /
`sgl-kernel` / `vllm` 这种 GPU-only 的——**别装**，它们在 sglang import 阶段
不会触达。

#### 6d. `import sglang` 时报 `cannot find libcudart.so` 或 CUDA 类错误

说明这个 commit 的 sglang 在 import 阶段就 eager-load 了 CUDA 库。退到稍老的
upstream commit 或用 PyPI 老版本，比如 `sglang==0.5.4`（SpecForge
`requirements-rocm.txt` 钉的版本）：

```bash
pip install sglang==0.5.4 --no-deps -i ...
```

---

## 装完之后

1. 每次开新 shell，重新 source CANN（步骤 0 那一坨）。
2. 训练前激活 `transfer_to_npu` shim。shim 文件和封装好的启动脚本会在
   upstream-strategy 计划的 PR3 里加上；在那之前最小激活方式见
   `docs/ascend_npu/upstream_strategy_zh.md` §5.4。
3. 单机多 NPU 训练：用 `examples/run_qwen3_8b_dflash_online.sh`，加上
   `--target-model-backend hf --attention-backend sdpa`。多机扩展见
   `docs/ascend_npu/multi_node_training_zh.md`。
