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
    token: str = None,
):
    # Use provided token or fall back to HF_TOKEN environment variable
    import os
    token = token or os.environ.get("HF_TOKEN", None)
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

    # Try multiple dataset sources in order (trust_remote_code removed in datasets>=2.20)
    DATASET_CANDIDATES = [
        ("datasets-server", "vector-institute/SciCap", "train"),
        ("HF Hub",          "shunk031/SciCap",         "train"),
        ("HF Hub fallback", "jmhessel/newyorker_caption_contest", None),  # placeholder, see below
    ]

    ds = None
    for source_name, split in [
        ("vector-institute/SciCap",       "train"),
        ("shunk031/SciCap",               "train"),
        ("taesiri/arxiv-figures",         "train"),
    ]:
        try:
            print(f"   Trying: {source_name} ...")
            ds = load_dataset(source_name, split=split, token=token)
            print(f"   ✓ Loaded from {source_name}")
            break
        except Exception as e:
            print(f"   ✗ Failed ({e.__class__.__name__})")
            continue

    if ds is None:
        raise RuntimeError(
            "\n❌ Could not load any SciCap dataset from HuggingFace.\n"
            "Please check your internet connection or HF_TOKEN, then try:\n"
            "  huggingface-cli login"
        )

    print(f"   Raw dataset size: {len(ds):,} samples")
    print(f"   Columns: {ds.column_names}")

    # Determine caption column name (varies by dataset)
    caption_col = next(
        (c for c in ["caption", "caption_str", "text"] if c in ds.column_names),
        ds.column_names[0],
    )
    image_col = next(
        (c for c in ["image", "figure", "img"] if c in ds.column_names),
        None,
    )

    # Filter and collect records
    records = []
    for i, sample in enumerate(tqdm(ds, desc="Filtering samples")):
        # Unify caption field
        sample_unified = dict(sample)
        if caption_col != "caption":
            sample_unified["caption"] = sample.get(caption_col, "")

        if not filter_sample(sample_unified):
            continue

        # Save image to disk
        img = sample.get(image_col) if image_col else None
        if img is None or not isinstance(img, Image.Image):
            continue

        img_filename = f"{i:07d}.jpg"
        img_path = image_dir / img_filename

        if not img_path.exists():
            img.convert("RGB").save(img_path, quality=90)

        records.append({
            "image_path": str(img_path),
            "caption": sample_unified["caption"].strip(),
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
    parser.add_argument("--token", type=str, default=None,
                        help="HuggingFace token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    prepare_scicap(
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        token=args.token,
    )
