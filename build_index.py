"""
build_index.py

Build a FAISS vector index over all scientific figures in the dataset.
The index allows fast approximate nearest-neighbor search at inference time.

After indexing, figure embeddings are stored so retrieval is instant
without re-encoding images on every query.

Usage:
    python build_index.py \
        --adapter_path checkpoints/lora_r8/best_adapter \
        --data_path data/scicap_train.jsonl \
        --index_path index/sciclip.faiss
"""

import os
import json
import pickle
import argparse
import torch
import numpy as np
import faiss
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import CLIPProcessor
from PIL import Image

from models.lora_clip import LoRACLIP


def encode_all_images(
    model: LoRACLIP,
    records: list[dict],
    processor: CLIPProcessor,
    batch_size: int = 128,
    device: torch.device = None,
) -> tuple[np.ndarray, list[dict]]:
    """
    Encode all images in the dataset into embedding vectors.

    Returns:
        embeddings: (N, 512) float32 numpy array
        metadata:   list of dicts with image_path, caption, etc.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    all_embeddings = []
    valid_records = []

    for i in tqdm(range(0, len(records), batch_size), desc="Encoding images"):
        batch_records = records[i : i + batch_size]
        images = []
        batch_valid = []

        for rec in batch_records:
            try:
                img = Image.open(rec["image_path"]).convert("RGB")
                images.append(img)
                batch_valid.append(rec)
            except Exception as e:
                print(f"  Warning: Could not load {rec['image_path']}: {e}")

        if not images:
            continue

        inputs = processor(images=images, return_tensors="pt", padding=True)
        pixel_values = inputs["pixel_values"].to(device)

        with torch.no_grad():
            embs = model.encode_image(pixel_values)  # (B, 512)

        all_embeddings.append(embs.cpu().numpy())
        valid_records.extend(batch_valid)

    embeddings = np.concatenate(all_embeddings, axis=0).astype("float32")
    return embeddings, valid_records


def build_faiss_index(
    embeddings: np.ndarray,
    use_gpu: bool = False,
) -> faiss.Index:
    """
    Build a FAISS IVFFlat index for efficient approximate nearest-neighbor search.

    IVFFlat:
        - Partitions the embedding space into n_lists clusters (Voronoi cells)
        - At query time, only searches n_probe nearest clusters
        - Good speed/accuracy tradeoff for 10k-1M vectors

    For small datasets (<10k), uses exact FlatIP (inner product) search.
    """
    n, d = embeddings.shape
    print(f"Building FAISS index: {n} vectors of dimension {d}")

    if n < 10_000:
        # Exact search for small datasets
        print("  Using exact FlatIP index (dataset < 10k)")
        index = faiss.IndexFlatIP(d)
    else:
        # Approximate search for larger datasets
        n_lists = min(int(np.sqrt(n)), 256)
        print(f"  Using IVFFlat index with {n_lists} clusters")
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, n_lists, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
        index.nprobe = min(32, n_lists)  # Search 32 clusters per query

    if use_gpu and faiss.get_num_gpus() > 0:
        index = faiss.index_cpu_to_all_gpus(index)
        print("  Moved index to GPU")

    index.add(embeddings)
    print(f"  ✓ Index built with {index.ntotal} vectors")
    return index


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print("\nLoading LoRA-CLIP model...")
    model = LoRACLIP.from_pretrained_lora(args.base_model, args.adapter_path)
    model = model.to(device)
    processor = CLIPProcessor.from_pretrained(args.base_model)

    # Load dataset records
    print(f"\nLoading records from {args.data_path}...")
    with open(args.data_path) as f:
        records = [json.loads(line) for line in f]
    print(f"  {len(records):,} records loaded")

    # Encode all images
    embeddings, valid_records = encode_all_images(
        model, records, processor, batch_size=args.batch_size, device=device
    )
    print(f"\n✓ Encoded {len(valid_records):,} images → shape {embeddings.shape}")

    # Build FAISS index
    index = build_faiss_index(embeddings)

    # Save index and metadata
    index_path = Path(args.index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(index_path))
    print(f"\n✓ FAISS index saved to: {index_path}")

    metadata_path = index_path.parent / "metadata.pkl"
    with open(metadata_path, "wb") as f:
        pickle.dump(valid_records, f)
    print(f"✓ Metadata saved to: {metadata_path}")

    # Save embeddings for offline analysis
    emb_path = index_path.parent / "embeddings.npy"
    np.save(str(emb_path), embeddings)
    print(f"✓ Embeddings saved to: {emb_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", type=str,
                        default="checkpoints/lora_r8/best_adapter")
    parser.add_argument("--data_path", type=str, default="data/scicap_train.jsonl")
    parser.add_argument("--index_path", type=str, default="index/sciclip.faiss")
    parser.add_argument("--base_model", type=str,
                        default="openai/clip-vit-base-patch32")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    main(args)
