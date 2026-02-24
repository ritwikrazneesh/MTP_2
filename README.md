# RemoteCLIP + Class-wise DualPrompt (Few-shot)

Implements a **DualPrompt-inspired** method for remote sensing few-shot classification:

- Backbone: **RemoteCLIP** (OpenCLIP-compatible weights)
- Encoders: **frozen** (vision + text)
- Prompts:
  - **G-prompt** shared (in early layers) for both encoders
  - **E-prompt per class** (in deeper layers) for **text encoder**
- Few-shot protocol: supervised **K-shot per class** (default `K=4`), test on remaining samples.

## Install

```bash
pip install -r requirements.txt
```

## Kaggle paths

RemoteCLIP checkpoint (example):

- `/kaggle/input/remoteclip-vitb32-pt/RemoteCLIP-ViT-B-32.pt`

Datasets are expected as ImageFolder-style directories:

### AID
`/kaggle/input/DATASETS/AID/AID/<Class>/*`

### EuroSAT RGB
`/kaggle/input/DATASETS/eurosat_rgb/2750/<Class>/*`

### PatternNet
`/kaggle/input/DATASETS/PatternNet/PatternNet/<Class>/*`

### UCMerced
`/kaggle/input/DATASETS/UCMerced LandUse Dataset/UCMerced_LandUse/<Class>/*`

## Run (example)

AID 4-shot:

```bash
python train_fewshot.py \
  --dataset aid \
  --data_root "/kaggle/input/DATASETS/AID/AID" \
  --k_shot 4 \
  --seed 0 \
  --remoteclip_ckpt "/kaggle/input/remoteclip-vitb32-pt/RemoteCLIP-ViT-B-32.pt" \
  --epochs 50 \
  --batch_size 64 \
  --out_dir "./outputs" \
  --run_name "aid_vitb32_k4_seed0"
```

## Notes / Next steps

- Current implementation uses **layerwise prefix tokens** (robust across OpenCLIP versions).
- If you later want strict **prefix-KV tuning**, we can vendor a fixed OpenCLIP attention implementation.
- We can also add:
  - multiple seeds and averaged reporting
  - official train/test split handling for NWPU-RESISC45 and WHU-RS19
  - CoCoOp-style context optimization / prompt ensembling
