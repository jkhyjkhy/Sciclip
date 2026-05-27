"""
evaluate.py

Evaluation utilities for SciCLIP.
Computes text→image retrieval metrics: Recall@k and MRR.

Usage (standalone):
    python evaluate.py \
        --adapter_path checkpoints/lora_r8/best_adapter \
        --val_path data/scicap_val.jsonl
"""

import argparse
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import CLIPProcessor

from models.lora_clip import LoRACLIP
from data.prepare_scicap import SciCapDataset


def compute_recall_at_k(
    query_embs: torch.Tensor,
    gallery_embs: torch.Tensor,
    ks: list[int] = [1, 5, 10],
) -> dict:
    """
    Compute Recall@k for text→image retrieval.

    Assumes query_embs[i] should retrieve gallery_embs[i] (1-to-1 pairing).
    This is the standard evaluation setup for image-text retrieval.

    Args:
        query_embs:   (N, D) normalized text embeddings
        gallery_embs: (N, D) normalized image embeddings
        ks: list of k values to evaluate

    Returns:
        dict with keys "recall@k" for each k, and "mrr"
    """
    n = query_embs.shape[0]

    # Compute full similarity matrix: (N, N)
    # sim[i, j] = cosine similarity between query i and gallery image j
    sim_matrix = query_embs @ gallery_embs.T  # (N, N)

    # For each query, rank gallery images by descending similarity
    # ranks[i] = rank of the ground-truth match for query i (0-indexed)
    sorted_indices = torch.argsort(sim_matrix, dim=1, descending=True)  # (N, N)
    ground_truth = torch.arange(n).unsqueeze(1)  # (N, 1)

    # Find rank of ground truth match for each query
    ranks = (sorted_indices == ground_truth).nonzero(as_tuple=False)[:, 1]  # (N,)

    metrics = {}

    # Recall@k: fraction of queries where ground truth is in Top-k
    for k in ks:
        recall = (ranks < k).float().mean().item()
        metrics[f"recall@{k}"] = recall

    # MRR: Mean Reciprocal Rank
    mrr = (1.0 / (ranks.float() + 1)).mean().item()
    metrics["mrr"] = mrr

    return metrics


def compute_median_rank(
    query_embs: torch.Tensor,
    gallery_embs: torch.Tensor,
) -> float:
    """Compute median rank of ground truth (lower is better)."""
    sim_matrix = query_embs @ gallery_embs.T
    sorted_indices = torch.argsort(sim_matrix, dim=1, descending=True)
    ground_truth = torch.arange(query_embs.shape[0]).unsqueeze(1)
    ranks = (sorted_indices == ground_truth).nonzero(as_tuple=False)[:, 1]
    return ranks.float().median().item()


@torch.no_grad()
def evaluate_model(
    model: LoRACLIP,
    dataloader: DataLoader,
    device: torch.device,
    ks: list[int] = [1, 5, 10],
) -> dict:
    """
    Run full evaluation: encode all samples, compute retrieval metrics.

    Returns dict with recall@k and mrr values.
    """
    model.eval()
    all_img_embs = []
    all_txt_embs = []

    for batch in tqdm(dataloader, desc="Encoding"):
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        img_emb = model.encode_image(pixel_values)
        txt_emb = model.encode_text(input_ids, attention_mask)

        all_img_embs.append(img_emb.cpu())
        all_txt_embs.append(txt_emb.cpu())

    all_img_embs = torch.cat(all_img_embs, dim=0)
    all_txt_embs = torch.cat(all_txt_embs, dim=0)

    metrics = compute_recall_at_k(all_txt_embs, all_img_embs, ks=ks)
    metrics["median_rank"] = compute_median_rank(all_txt_embs, all_img_embs)

    return metrics


def compare_baseline_vs_lora(
    adapter_path: str,
    val_path: str,
    base_model: str = "openai/clip-vit-base-patch32",
    batch_size: int = 64,
):
    """
    Compare vanilla CLIP vs LoRA-CLIP on the validation set.
    Prints a side-by-side metrics table for the term paper.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = CLIPProcessor.from_pretrained(base_model)

    val_ds = SciCapDataset(val_path, processor)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    results = {}

    # --- Baseline: vanilla CLIP (no LoRA) ---
    print("\nEvaluating vanilla CLIP (baseline)...")
    baseline = LoRACLIP(model_name=base_model)
    # Disable LoRA adapters by merging with zero weights (effectively vanilla)
    baseline = baseline.to(device)
    results["Vanilla CLIP"] = evaluate_model(baseline, val_loader, device)
    del baseline

    # --- LoRA-CLIP ---
    print("\nEvaluating LoRA-CLIP...")
    lora_model = LoRACLIP.from_pretrained_lora(base_model, adapter_path)
    lora_model = lora_model.to(device)
    results["LoRA-CLIP"] = evaluate_model(lora_model, val_loader, device)

    # Print comparison table
    print("\n" + "=" * 60)
    print(f"{'Model':<20} {'R@1':>8} {'R@5':>8} {'R@10':>8} {'MRR':>8} {'Med.Rank':>10}")
    print("-" * 60)
    for model_name, metrics in results.items():
        print(
            f"{model_name:<20} "
            f"{metrics['recall@1']:>8.4f} "
            f"{metrics['recall@5']:>8.4f} "
            f"{metrics['recall@10']:>8.4f} "
            f"{metrics['mrr']:>8.4f} "
            f"{metrics['median_rank']:>10.1f}"
        )
    print("=" * 60)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", type=str, required=True,
                        help="Path to saved LoRA adapter (e.g. checkpoints/lora_r8/best_adapter)")
    parser.add_argument("--val_path", type=str, default="data/scicap_val.jsonl")
    parser.add_argument("--base_model", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    compare_baseline_vs_lora(
        adapter_path=args.adapter_path,
        val_path=args.val_path,
        base_model=args.base_model,
        batch_size=args.batch_size,
    )
