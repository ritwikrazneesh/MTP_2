from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from typing import Dict

import torch

from rcdp.backbones.remoteclip import RemoteCLIPConfig, load_remoteclip, freeze_remoteclip
from rcdp.data.datasets import DatasetSpec, build_dataset
from rcdp.data.apply_transform import TransformDataset
from rcdp.data.loaders import LoaderConfig, build_fewshot_loaders, describe_split
from rcdp.data.transforms import build_transforms
from rcdp.models.dualprompt import DualPromptConfig, RemoteCLIPDualPromptModel
from rcdp.train.loops import TrainConfig, train_fewshot
from rcdp.utils.device import get_device
from rcdp.utils.labels import canonicalize_classname
from rcdp.utils.seed import SeedConfig, seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("RemoteCLIP + Class-wise DualPrompt (few-shot)")

    # Dataset
    p.add_argument("--dataset", type=str, required=True,
                   choices=["aid", "eurosat_rgb", "patternnet", "ucm", "nwpu-resisc45", "whu-rs19"])
    p.add_argument("--data_root", type=str, required=True, help="Path to dataset root containing class folders")

    # Few-shot
    p.add_argument("--k_shot", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)

    # RemoteCLIP
    p.add_argument("--model_name", type=str, default="ViT-B-32")
    p.add_argument("--remoteclip_ckpt", type=str, default="/kaggle/input/remoteclip-vitb32-pt/RemoteCLIP-ViT-B-32.pt")

    # DualPrompt
    p.add_argument("--template", type=str, default="a satellite image of {}.")
    p.add_argument("--prefix_len", type=int, default=5)
    p.add_argument("--g_layers", type=int, default=6)
    p.add_argument("--e_layers", type=int, default=6)

    # Train
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--device", type=str, default="cuda")

    # Output
    p.add_argument("--out_dir", type=str, default="./outputs")
    p.add_argument("--run_name", type=str, default="run")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(SeedConfig(seed=args.seed, deterministic=True))

    device = get_device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    # --- load backbone ---
    bundle = load_remoteclip(RemoteCLIPConfig(
        model_name=args.model_name,
        checkpoint_path=args.remoteclip_ckpt,
        device=str(device),
        precision="amp" if not args.no_amp else "fp32",
    ))
    freeze_remoteclip(bundle.model)

    # --- dataset ---
    dataset, classnames_raw = build_dataset(DatasetSpec(name=args.dataset, root=args.data_root))
    classnames = [canonicalize_classname(c) for c in classnames_raw]

    # --- transforms ---
    tfm = build_transforms(bundle.preprocess_train, bundle.preprocess_val)
    dataset_trainable = TransformDataset(dataset, transform=tfm.train)  # we will split after transform wrapper

    # We need targets from original dataset; indices still align
    train_loader, test_loader, split = build_fewshot_loaders(
        dataset_trainable,
        k_shot=args.k_shot,
        seed=args.seed,
        cfg=LoaderConfig(batch_size=args.batch_size, num_workers=args.num_workers),
    )

    print("[split]\n" + describe_split(dataset, split, classnames_raw))

    # --- model ---
    dp_cfg = DualPromptConfig(
        template=args.template,
        prefix_len=args.prefix_len,
        g_layers=args.g_layers,
        e_layers=args.e_layers,
    )
    model = RemoteCLIPDualPromptModel(
        remoteclip_model=bundle.model,
        tokenizer=bundle.tokenizer,
        classnames=classnames,
        cfg=dp_cfg,
        device=device,
    ).to(device)

    # --- train ---
    results = train_fewshot(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        cfg=TrainConfig(
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            use_amp=(not args.no_amp),
        ),
        device=device,
    )

    # --- save ---
    run_dir = os.path.join(args.out_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    ckpt_path = os.path.join(run_dir, "dualprompt_prefix.pt")
    torch.save({
        "dualprompt_config": asdict(dp_cfg),
        "classnames": classnames,
        "state_dict": model.state_dict(),
    }, ckpt_path)

    res_path = os.path.join(run_dir, "results.json")
    payload: Dict = {
        "args": vars(args),
        "dualprompt_config": asdict(dp_cfg),
        "results": results,
        "checkpoint": ckpt_path,
    }
    with open(res_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"[done] saved checkpoint: {ckpt_path}")
    print(f"[done] saved results:    {res_path}")
    print(f"[done] best_test_acc:    {results['best_test_acc']:.4f}")


if __name__ == "__main__":
    main()
