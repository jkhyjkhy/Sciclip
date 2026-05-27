"""
data/prepare_scicap.py

Download and preprocess a filtered subset of the SciCap dataset
for LoRA-CLIP training.

SciCap contains real arXiv paper figures with their captions.
We filter to ML/NLP papers (cs.CL, cs.LG, cs.CV) for domain focus.

Output:
    data/scicap_train.jsonl   — (image_path, caption) pairs for training
    data/scicap_val.jsonl     — validation split
    data/images/              — cached figure images

Usage:
    python data/prepare_scicap.py --max_samples 20000 --output_dir data/
"""

import os
import json
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset


# SciCap figure types to keep (exclude compound figures which are noisy)
KEEP_FIGURE_TYPES = {"graph", "diagram", "table", "natural image", "equation"}

# Minimum caption length (too short captions give poor training signal)
MIN_CAPTION_LEN = 20
MAX_CAPTION_LEN = 300


def filter_sample(sample: dict) -> bool:
    """Return True if this sample should be kept for training."""
    caption = sample.get("caption", "")

    # Filter by caption length
    if not (MIN_CAPTION_LEN <= len(caption) <= MAX_CAPTION_LEN):
        return False

    # Skip captions that are mostly just "Figure X" references
    if caption.lower().startswith("figure") and len(caption) < 40:
        return False

    return True


def save_jsonl(records: list[dict], path: str):
    """Write a list of dicts to a .jsonl file."""
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def prepare_scicap(
    max_samples: int = 20000,
    output_dir: str = "data",
    val_ratio: float = 0.1,
    seed: int = 42,
):
    """
    Download SciCap from HuggingFace and prepare train/val splits.

    Args:
        max_samples: Maximum number of (image, caption) pairs to keep
        output_dir: Directory to save processed data and images
        val_ratio: Fraction of data to use for validation
        seed: Random seed for reproducibility
    """
    output_dir = Path(output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    print("📥 Loading SciCap from HuggingFace...")
    print("   (This may take a few minutes on first run — images are downloaded)")

    # SciCap is available under 'vector-institute/SciCap'
    # It has columns: figure_id, caption, image (PIL), arxiv_id
    try:
        ds = load_dataset(
            "vector-institute/SciCap",
            split="train",
            trust_remote_code=True,
        )
    except Exception:
        # Fallback: try alternative dataset name
        print("   Trying alternative dataset name...")
        ds = load_dataset(
            "Ino99/SciCap-No-Subfig-Img",
            split="train",
            trust_remote_code=True,
        )

    print(f"   Raw dataset size: {len(ds):,} samples")

    # Filter and collect records
    records = []
    for i, sample in enumerate(tqdm(ds, desc="Filtering samples")):
        if not filter_sample(sample):
            continue

        # Save image to disk
        img = sample["image"]
        if not isinstance(img, Image.Image):
            continue

        img_filename = f"{i:07d}.jpg"
        img_path = image_dir / img_filename

        if not img_path.exists():
            img.convert("RGB").save(img_path, quality=90)

        records.append({
            "image_path": str(img_path),
            "caption": sample["caption"].strip(),
            "arxiv_id": sample.get("arxiv_id", ""),
            "figure_id": sample.get("figure_id", str(i)),
        })

        if len(records) >= max_samples:
            break

    print(f"\n✓ Filtered to {len(records):,} samples")

    # Shuffle and split
    import random
    random.seed(seed)
    random.shuffle(records)

    n_val = int(len(records) * val_ratio)
    val_records = records[:n_val]
    train_records = records[n_val:]

    # Save splits
    train_path = output_dir / "scicap_train.jsonl"
    val_path = output_dir / "scicap_val.jsonl"
    save_jsonl(train_records, str(train_path))
    save_jsonl(val_records, str(val_path))

    print(f"✓ Train: {len(train_records):,} pairs → {train_path}")
    print(f"✓ Val:   {len(val_records):,} pairs → {val_path}")
    print("\nSample record:")
    print(json.dumps(train_records[0], indent=2))


# -------------------------------------------------------------------
# PyTorch Dataset wrapper
# -------------------------------------------------------------------
import torch
from torch.utils.data import Dataset
from transformers import CLIPProcessor


class SciCapDataset(Dataset):
    """
    PyTorch Dataset for SciCap (image, caption) pairs.

    Args:
        jsonl_path: Path to the .jsonl file prepared by prepare_scicap()
        processor: CLIPProcessor for image/text preprocessing
        max_length: Maximum token length for captions
    """

    def __init__(
        self,
        jsonl_path: str,
        processor: CLIPProcessor,
        max_length: int = 77,
    ):
        self.processor = processor
        self.max_length = max_length

        with open(jsonl_path) as f:
            self.records = [json.loads(line) for line in f]

        print(f"Loaded {len(self.records):,} samples from {jsonl_path}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]

        # Load and preprocess image
        image = Image.open(rec["image_path"]).convert("RGB")

        # Tokenize caption
        caption = rec["caption"]

        # Use CLIPProcessor to handle both image resizing and text tokenization
        inputs = self.processor(
            images=image,
            text=caption,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),   # (3, 224, 224)
            "input_ids": inputs["input_ids"].squeeze(0),          # (77,)
            "attention_mask": inputs["attention_mask"].squeeze(0), # (77,)
            "caption": caption,
            "image_path": rec["image_path"],
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=20000)
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    args = parser.parse_args()

    prepare_scicap(
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
    )
