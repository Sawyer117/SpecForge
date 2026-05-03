# SpecForge on Ascend NPU — Installation Guide (HF Backend)

> Reproducible install for SpecForge DFlash with the HF target-model backend
> on Ascend NPU (Atlas A2 / 910b / A3). Versions and the SpecForge commit
> are pinned in `requirements-ascend.txt` next to this file.
>
> If anything fails, copy the exact error and report it — see
> [Troubleshooting](#troubleshooting) for the five most common failure
> modes and what to try.

---

## What you get

After this install:
- SpecForge pinned to upstream commit `d5fb617f735db85a680876327f8173de0cb57c15`
- DFlash + HF target-model backend (no sglang, no NPU kernel build)
- yunchang `0.6.4` installed but dormant (DFlash default args do not exercise it)
- Ready to launch `scripts/train_dflash.py --target-model-backend hf --attention-backend sdpa`

What this install does **not** include (deliberately):
- `sglang` — only needed for `--target-model-backend sglang`
- `flash-attn` — has no NPU build
- `triton-ascend`, `BiSheng`, `sgl-kernel-npu`, `DeepEP` — only needed for the
  sglang backend's NPU kernels

---

## Prerequisites

- An existing **CANN ≥ 8.5.0** install on the host. Note its path; you will
  pass it as `CANN_HOME`.
- **Conda** (Miniconda/Anaconda) on `$PATH`.
- Network access to the Huawei Cloud PyPI mirror
  (`mirrors.huaweicloud.com`) and to GitHub (`github.com`).
- Python 3.11 will be installed by step 1; you do not need it system-wide.

---

## Install steps

Each step is independent. If one fails, fix it and re-run that step only.
The exact pinned versions referenced below are in
`docs/ascend_npu/requirements-ascend.txt`, which lives in the same branch
you'll clone in step 2.

### Step 0 — Source CANN

```bash
export CANN_HOME=/path/to/your/CANN/8.5.0.x       # ← edit this to your real CANN path
source "$CANN_HOME/ascend-toolkit/set_env.sh"
[ -f "$CANN_HOME/nnal/asdsip/set_env.sh" ] && source "$CANN_HOME/nnal/asdsip/set_env.sh"
[ -f "$CANN_HOME/nnal/atb/set_env.sh" ]    && source "$CANN_HOME/nnal/atb/set_env.sh"
```

### Step 1 — Create a conda environment (Python 3.11)

```bash
conda create -p ./conda/specforge_npu python=3.11 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ./conda/specforge_npu
```

### Step 2 — Clone the fork's `docs/ascend-npu` branch

This branch contains:
- SpecForge source at the pinned upstream commit `d5fb617`
- `docs/ascend_npu/` with this guide and `requirements-ascend.txt`

```bash
git clone -b docs/ascend-npu https://github.com/Sawyer117/SpecForge.git
cd SpecForge
```

### Step 3 — Install torch / torchvision / torch_npu from the Huawei Cloud mirror

```bash
pip install torch==2.9.0 torchvision==0.24.0 \
    -i https://mirrors.huaweicloud.com/repository/pypi/simple/ \
    --trusted-host mirrors.huaweicloud.com

pip install torch_npu==2.9.0 \
    -i https://mirrors.huaweicloud.com/repository/pypi/simple/ \
    --trusted-host mirrors.huaweicloud.com
```

> If `torch_npu==2.9.0` is not on the mirror, see
> [Troubleshooting #1](#1-torch_npu290-not-found-on-the-mirror).

### Step 4 — Install the rest of the pinned dependencies

```bash
# Strip the torch / torchvision lines (already installed in step 3)
sed -i '/^torch==/d; /^torchvision==/d' docs/ascend_npu/requirements-ascend.txt

pip install -r docs/ascend_npu/requirements-ascend.txt
```

> **Do not** undo the `sed`. After install, the file edit is local-only —
> when you `git status` you will see a modified file, which you can safely
> `git checkout -- docs/ascend_npu/requirements-ascend.txt` later.

### Step 5 — Editable-install SpecForge **without** re-resolving deps

```bash
pip install -e . --no-deps
```

`--no-deps` is **required**. Without it, pip would try to satisfy SpecForge's
own `pyproject.toml` pins (`torch==2.9.1`, `sglang==0.5.9`), both of which
conflict with what you intentionally installed in steps 3–4.

### Step 6 — Verify

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

### Expected output

```
torch                    : 2.9.0
torch_npu                : 2.9.0          (or 2.9.0.postN, exact value depends on the mirror)
transformers             : 4.57.1
yunchang.HAS_NPU         : True
yunchang.HAS_FLASH_ATTN  : False
torch.npu.is_available() : True
torch.npu.device_count() : 8              (your real NPU count)
specforge import OK
```

If all six lines look correct, the environment is ready for training.

---

## Troubleshooting

### 1. `torch_npu==2.9.0` not found on the mirror

Try without a strict pin first:

```bash
pip install torch_npu \
    -i https://mirrors.huaweicloud.com/repository/pypi/simple/ \
    --trusted-host mirrors.huaweicloud.com
```

Note the actual version pip picked (e.g. `torch_npu-2.9.0.post1`) and update
`requirements-ascend.txt` accordingly. If even that fails, the wheel is not
on the public Huawei Cloud mirror — ask your CANN team for the
release-channel URL or grab the wheel from
<https://www.hiascend.com/document/redirect/CannCommercialDeveloperResource>.

### 2. `numpy<2.0` conflicts with another package

If pip reports a resolver conflict on `numpy`, replace the loose pin with a
concrete version:

```bash
pip install numpy==1.26.4
```

Then re-run step 4 with `numpy` already pinned.

### 3. `setuptools<81` conflict

Same idea — pin to a concrete version:

```bash
pip install "setuptools==80.9.0"
```

`setuptools<81` is required because `torch_npu.dynamo.torchair` still imports
`pkg_resources`, which setuptools 81+ removed.

### 4. yunchang accidentally pulls flash-attn

Should not happen with `pip install yunchang==0.6.4` (no `[flash]` extra),
but if it does, install yunchang explicitly without extras:

```bash
pip install --no-deps yunchang==0.6.4
pip install "torch>=2.3.0"   # the only real yunchang dep
```

### 5. `from yunchang.globals import ...` raises `ImportError`

Capture the full traceback. Common cause is yunchang's `__init__.py` running
`from .ring import *`, which transitively imports a submodule that needs
something missing on the system. Report the traceback — yunchang 0.6.4 is
expected to be NPU-clean, so any failure here is genuinely interesting.

---

## Next steps after install

1. Source the ATB runtime once per shell (already done in step 0 if you ran
   the full block).
2. Activate the `transfer_to_npu` shim and launch training. The shim and a
   ready-made launch script will land in PR3 of the upstream-strategy plan;
   until then, see `docs/ascend_npu/upstream_strategy.md` §5.4 for the
   minimal activation pattern.
3. Single-node multi-NPU training: `examples/run_qwen3_8b_dflash_online.sh`
   with `--target-model-backend hf --attention-backend sdpa`. For multi-node
   see `docs/ascend_npu/multi_node_training.md`.
