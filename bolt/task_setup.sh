#!/usr/bin/env bash
# Environment setup for the GestureLens text-ablation Bolt job.
#
# Kept deliberately small: the ablation needs torch, transformers and the usual
# scientific stack, nothing else. There is no conda step because this repo has
# no compiled extensions and no package to `pip install -e`.
set -eux

# Pin the CUDA wheel index. Installing unpinned "latest" torch pulls CUDA 13.0
# wheels, which fail on a CUDA 12.8 cluster driver with
# "The NVIDIA driver on your system is too old". cu126 is the compatible build.
python -m pip install --upgrade pip
# torchvision comes from the SAME index as torch. Installing it later from PyPI
# would resolve a torchvision built against a different torch and silently
# reinstall torch with the wrong CUDA wheels.
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# transformers is only needed by the three BERT-backed arms (real, shuffled,
# mismatched); onehot/random/none run without it. The Tier 0 job does not need
# it at all -- it ships precomputed similarity matrices instead.
# numpy is left unpinned: the sweep was verified locally on numpy 2.0.2, and the
# repo's one numpy-2.0 incompatibility (np.bool in utils.match_labels) is fixed.
#
# torchvision / tqdm / tensorboard are NOT ours: they are transitive imports of
# ContrastSense-Public, whose ContrastSense.py we import to get the real
# encoder class. Determined by walking that repo's whole import chain
# (ContrastSense -> utils -> data_loader -> data_preprocessing), so this should
# be the complete set. tensorboard hides behind `from torch.utils.tensorboard
# import SummaryWriter`, which needs the standalone package.
python -m pip install \
  "transformers>=4.30" \
  numpy \
  pandas \
  scikit-learn \
  scipy \
  matplotlib \
  seaborn \
  pyyaml \
  tqdm \
  tensorboard \
  h5py

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY
