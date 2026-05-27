"""
train.py

Training script for SciCLIP: LoRA-adapted CLIP on scientific figures.

Pipeline:
    1. Load SciCap train/val splits
    2. Initialize LoRA-CLIP model
    3. Train with symmetric InfoNCE contrastive loss
    4. Evaluate Recall@k on validation set after each epoch
    5. Save best LoRA adapter checkpoint

Usage:
    # Basic run
    python train.py

    # With custom settings
    python train.py --lora_r 16 --epochs 5 --batch_size 64 --wandb

    # Ablation study (runs r=4,8,16 sequentially)
    python train.py --ablation
"""

import os
import json
import argparse
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import CLIPProcessor
from tqdm import tqdm

from models.lora_clip import LoRACLIP, contrastive_loss, DEFAULT_LORA_CONFIG
from data.prepare_scicap import SciCapDataset
from evaluate import compute_recall_at_k
from peft import LoraConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Train SciCLIP")

    # LoRA hyperparameters
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA rank r (try 4, 8, 16 for ablation)")
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=100)

    # Data
    parser.add_argument("--train_path", type=str, default="data/scicap_train.jsonl")
    parser.add_argument("--val_path", type=str, default="data/scicap_val.jsonl")
    parser.add_argument("--num_workers", type=int, default=4)

    # Experiment
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--base_model", type=str,
                        default="openai/clip-vit-base-patch32")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--ablation", action="store_true",
                        help="Run ablation over lora_r in [4, 8, 16]")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")   # Apple Silicon
    return torch.device("cpu")


def train_one_epoch(
    model: LoRACLIP,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
) -> float:
    """Run one training epoch, return average loss."""
    model.train()
    total_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [train]")
    for batch in pbar:
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        optimizer.zero_grad()

        img_emb, txt_emb, logit_scale = model(
            pixel_values, input_ids, attention_mask
        )
        loss = contrastive_loss(img_emb, txt_emb, logit_scale)

        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(dataloader)


@torch.no_grad()
def validate(
    model: LoRACLIP,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    """
    Compute validation loss and Recall@k metrics.

    We encode all validation (image, text) pairs, then measure how often
    the correct image appears in the Top-k results for a text query.
    """
    model.eval()
    all_img_embs = []
    all_txt_embs = []
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Validating"):
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        img_emb, txt_emb, logit_scale = model(
            pixel_values, input_ids, attention_mask
        )
        loss = contrastive_loss(img_emb, txt_emb, logit_scale)
        total_loss += loss.item()

        all_img_embs.append(img_emb.cpu())
        all_txt_embs.append(txt_emb.cpu())

    # Stack all embeddings
    all_img_embs = torch.cat(all_img_embs, dim=0)  # (N, 512)
    all_txt_embs = torch.cat(all_txt_embs, dim=0)  # (N, 512)

    # Compute Recall@k for text→image retrieval
    recall_metrics = compute_recall_at_k(
        query_embs=all_txt_embs,
        gallery_embs=all_img_embs,
        ks=[1, 5, 10],
    )

    recall_metrics["val_loss"] = total_loss / len(dataloader)
    return recall_metrics


def run_training(args, lora_r: int = None):
    """Main training loop for a single LoRA rank configuration."""
    if lora_r is None:
        lora_r = args.lora_r

    set_seed(args.seed)
    device = get_device()
    print(f"\n{'='*50}")
    print(f"Training SciCLIP | LoRA r={lora_r} | Device: {device}")
    print(f"{'='*50}\n")

    # ------- Weights & Biases -------
    if args.wandb:
        import wandb
        wandb.init(
            project="sciclip",
            name=f"lora_r{lora_r}",
            config={"lora_r": lora_r, **vars(args)},
        )

    # ------- Model -------
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=args.lora_alpha,
        # Simple layer names — PEFT finds all q/k/v projections in both encoders.
        # Wildcard paths (e.g. 'encoder.layers.*.q_proj') not supported in PEFT>=0.9
        target_modules=["q_proj", "k_proj", "v_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    model = LoRACLIP(
        model_name=args.base_model,
        lora_config=lora_config,
    ).to(device)
    model.print_trainable_parameters()

    # ------- Data -------
    processor = CLIPProcessor.from_pretrained(args.base_model)

    train_ds = SciCapDataset(args.train_path, processor)
    val_ds = SciCapDataset(args.val_path, processor)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ------- Optimizer & Scheduler -------
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps - args.warmup_steps
    )

    # ------- Training Loop -------
    output_dir = Path(args.output_dir) / f"lora_r{lora_r}"
    output_dir.mkdir(parents=True, exist_ok=True)

    best_recall = 0.0
    history = []

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch
        )
        metrics = validate(model, val_loader, device)

        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {metrics['val_loss']:.4f}")
        print(f"  R@1:  {metrics['recall@1']:.4f}")
        print(f"  R@5:  {metrics['recall@5']:.4f}")
        print(f"  R@10: {metrics['recall@10']:.4f}")

        history.append({"epoch": epoch + 1, "train_loss": train_loss, **metrics})

        if args.wandb:
            import wandb
            wandb.log({"train_loss": train_loss, **metrics, "epoch": epoch + 1})

        # Save best checkpoint
        if metrics["recall@5"] > best_recall:
            best_recall = metrics["recall@5"]
            model.save_lora_adapter(str(output_dir / "best_adapter"))
            print(f"  ✓ New best R@5={best_recall:.4f} — adapter saved")

    # Save training history
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n✓ Training done! Best R@5: {best_recall:.4f}")
    return best_recall


if __name__ == "__main__":
    args = parse_args()

    if args.ablation:
        # Run ablation over LoRA ranks
        results = {}
        for r in [4, 8, 16]:
            best = run_training(args, lora_r=r)
            results[f"r={r}"] = best

        print("\n" + "=" * 40)
        print("Ablation Results (Best R@5):")
        for k, v in results.items():
            print(f"  {k}: {v:.4f}")
    else:
        run_training(args)
