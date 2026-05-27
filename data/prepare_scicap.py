"""
data/prepare_scicap.py

Download and preprocess SciCap (CrowdAILab/scicap) for LoRA-CLIP training.

Dataset structure (COCO format):
    train.json     → {"images": [...], "annotations": [...]}
    img-split.zip  → all training figure images (PNG files)

We use hf_hub_download() to get these files directly, bypassing
load_dataset() which fails due to column mismatches across JSON files.

Caption field: 'caption_no_index' (removes "Figure X:" prefix — better for retrieval)

Usage:
    python data/prepare_scicap.py --max_samples 200   # quick test
    python data/prepare_scicap.py --max_samples 20000 # full training

⚠️  img-split.zip can be 10-20 GB. It is cached in ~/.cache/huggingface/
    after the first download — subsequent runs are fast.
"""

import os
import json
import argparse
import random
import zipfile
from pathlib import Path
from PIL import Image
from tqdm import tqdm


# Caption quality filters
MIN_CAPTION_LEN = 20
MAX_CAPTION_LEN = 300


def filter_sample(sample: dict) -> bool:
    """Return True if this caption is worth training on."""
    caption = sample.get("caption", "")
    if not (MIN_CAPTION_LEN <= len(caption) <= MAX_CAPTION_LEN):
        return False
    # Skip bare "Figure X" captions with no real content
    if caption.lower().startswith("figure") and len(caption) < 40:
        return False
    return True


def save_jsonl(records: list, path: str):
    """Write list of dicts to a .jsonl file."""
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
    """
    Download SciCap and prepare train/val splits.

    Args:
        max_samples : max (image, caption) pairs to use
        output_dir  : where to save .jsonl files and extracted images
        val_ratio   : fraction of data to hold out for validation
        seed        : random seed
        token       : HuggingFace token (or set HF_TOKEN env var)
    """
    from huggingface_hub import hf_hub_download

    token = token or os.environ.get("HF_TOKEN", None)

    output_dir = Path(output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Step 1 — Download train.json (COCO format, ~882 MB)                 #
    # ------------------------------------------------------------------ #
    print("=" * 55)
    print("📥 SciCLIP Data Preparation")
    print("=" * 55)
    print("\n[1/3] Downloading train.json (~882 MB, cached after first run)…")

    json_path = hf_hub_download(
        repo_id="CrowdAILab/scicap",
        filename="train.json",
        repo_type="dataset",
        token=token,
    )

    print("      Parsing COCO JSON…")
    with open(json_path) as f:
        coco = json.load(f)

    # id → file_name mapping
    id_to_filename = {img["id"]: img["file_name"] for img in coco["images"]}
    print(f"      Total images in dataset : {len(id_to_filename):,}")

    # Filter annotations by caption quality
    # caption_no_index = caption with "Figure X:" prefix removed → better retrieval signal
    valid_anns = []
    for ann in coco["annotations"]:
        caption = ann.get("caption_no_index", ann.get("caption", "")).strip()
        if filter_sample({"caption": caption}) and ann["image_id"] in id_to_filename:
            valid_anns.append({
                "image_id": ann["image_id"],
                "caption": caption,
            })

    print(f"      Annotations passing filter: {len(valid_anns):,}")

    # Shuffle and pick max_samples
    random.seed(seed)
    random.shuffle(valid_anns)
    selected_anns = valid_anns[:max_samples]
    needed_filenames = {id_to_filename[a["image_id"]] for a in selected_anns}
    print(f"      Selecting {len(needed_filenames):,} images to extract")

    # ------------------------------------------------------------------ #
    # Step 2 — Download img-split.zip (training images)                  #
    # ------------------------------------------------------------------ #
    print("\n[2/3] Downloading img-split.zip…")
    print("      ⚠️  This file can be 10–20 GB. Download may take 15–30 min.")
    print("      It is cached in ~/.cache/huggingface/ — only downloaded once.")

    zip_path = hf_hub_download(
        repo_id="CrowdAILab/scicap",
        filename="img-split.zip",
        repo_type="dataset",
        token=token,
    )
    print(f"      Cached at: {zip_path}")

    # ------------------------------------------------------------------ #
    # Step 3 — Selectively extract only the images we need               #
    # ------------------------------------------------------------------ #
    print(f"\n[3/3] Extracting {len(needed_filenames):,} images from zip…")

    extracted_map: dict[str, Path] = {}

    with zipfile.ZipFile(zip_path, "r") as zf:
        # Build basename → full zip entry mapping (entries may have subdir prefix)
        basename_to_entry = {}
        for entry in zf.namelist():
            base = os.path.basename(entry)
            if base:
                basename_to_entry[base] = entry

        for fname in tqdm(needed_filenames, desc="Extracting"):
            out_path = image_dir / fname
            if out_path.exists():
                extracted_map[fname] = out_path
                continue

            entry = basename_to_entry.get(fname)
            if entry is None:
                continue  # image not found in zip

            data = zf.read(entry)
            with open(out_path, "wb") as f:
                f.write(data)
            extracted_map[fname] = out_path

    print(f"      Extracted {len(extracted_map):,} images")

    # ------------------------------------------------------------------ #
    # Build final records                                                  #
    # ------------------------------------------------------------------ #
    records = []
    for ann in selected_anns:
        fname = id_to_filename.get(ann["image_id"])
        if not fname or fname not in extracted_map:
            continue

        img_path = extracted_map[fname]

        # Verify image can be opened
        try:
            with Image.open(img_path) as img:
                img.verify()
        except Exception:
            continue

        records.append({
            "image_path": str(img_path),
            "caption": ann["caption"],
            "arxiv_id": "",
            "figure_id": str(ann["image_id"]),
        })

    print(f"\n✓ {len(records):,} valid (image, caption) pairs collected")

    # Shuffle and split train / val
    random.seed(seed)
    random.shuffle(records)

    n_val = max(1, int(len(records) * val_ratio))
    val_records   = records[:n_val]
    train_records = records[n_val:]

    train_path = output_dir / "scicap_train.jsonl"
    val_path   = output_dir / "scicap_val.jsonl"
    save_jsonl(train_records, str(train_path))
    save_jsonl(val_records,   str(val_path))

    print(f"✓ Train : {len(train_records):,} pairs → {train_path}")
    print(f"✓ Val   : {len(val_records):,} pairs → {val_path}")
    if train_records:
        print(f"\nSample record:\n{json.dumps(train_records[0], indent=2)}")


# ------------------------------------------------------------------ #
# PyTorch Dataset wrapper                                             #
# ------------------------------------------------------------------ #
import torch
from torch.utils.data import Dataset
from transformers import CLIPProcessor


class SciCapDataset(Dataset):
    """
    PyTorch Dataset for SciCap (image, caption) pairs.

    Args:
        jsonl_path : path to .jsonl file from prepare_scicap()
        processor  : CLIPProcessor for image/text preprocessing
        max_length : max token length for captions (CLIP default: 77)
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
        image = Image.open(rec["image_path"]).convert("RGB")
        caption = rec["caption"]

        inputs = self.processor(
            images=image,
            text=caption,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        return {
            "pixel_values":   inputs["pixel_values"].squeeze(0),    # (3, 224, 224)
            "input_ids":      inputs["input_ids"].squeeze(0),        # (77,)
            "attention_mask": inputs["attention_mask"].squeeze(0),   # (77,)
            "caption":        caption,
            "image_path":     rec["image_path"],
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare SciCap data for SciCLIP")
    parser.add_argument("--max_samples", type=int, default=20000)
    parser.add_argument("--output_dir",  type=str, default="data")
    parser.add_argument("--val_ratio",   type=float, default=0.1)
    parser.add_argument("--token",       type=str, default=None,
                        help="HuggingFace token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    prepare_scicap(
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        token=args.token,
    )
