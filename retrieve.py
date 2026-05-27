"""
retrieve.py

Command-line interface for querying the SciCLIP FAISS index.
Encodes a text query with LoRA-CLIP and returns the Top-k
most similar scientific figures.

Usage:
    python retrieve.py --query "attention mechanism transformer architecture"
    python retrieve.py --query "loss curve training validation" --top_k 10
    python retrieve.py --interactive   # interactive mode
"""

import argparse
import pickle
import json
import torch
import faiss
import numpy as np
from pathlib import Path
from transformers import CLIPProcessor

from models.lora_clip import LoRACLIP


class SciCLIPRetriever:
    """
    Wrapper for fast text→figure retrieval using a pre-built FAISS index.

    Usage:
        retriever = SciCLIPRetriever(adapter_path, index_path)
        results = retriever.search("attention mechanism diagram", top_k=5)
    """

    def __init__(
        self,
        adapter_path: str,
        index_path: str,
        base_model: str = "openai/clip-vit-base-patch32",
        device: str = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        print("Loading LoRA-CLIP model...")
        self.model = LoRACLIP.from_pretrained_lora(base_model, adapter_path)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.processor = CLIPProcessor.from_pretrained(base_model)

        # Load FAISS index
        index_path = Path(index_path)
        print(f"Loading FAISS index from {index_path}...")
        self.index = faiss.read_index(str(index_path))

        # Load figure metadata (image paths, captions)
        metadata_path = index_path.parent / "metadata.pkl"
        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)

        print(f"✓ Ready — {self.index.ntotal:,} figures indexed")

    @torch.no_grad()
    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Search for the top-k most similar figures for a text query.

        Args:
            query: Natural language description of the figure to find
            top_k: Number of results to return

        Returns:
            List of dicts with keys: rank, score, image_path, caption, arxiv_id
        """
        # Encode query text
        inputs = self.processor(
            text=query,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        query_emb = self.model.encode_text(input_ids, attention_mask)
        query_vec = query_emb.cpu().numpy().astype("float32")

        # FAISS search
        scores, indices = self.index.search(query_vec, top_k)

        # Collect results
        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:
                continue
            rec = self.metadata[idx]
            results.append({
                "rank": rank + 1,
                "score": float(score),
                "image_path": rec["image_path"],
                "caption": rec["caption"],
                "arxiv_id": rec.get("arxiv_id", ""),
                "figure_id": rec.get("figure_id", ""),
            })

        return results


def print_results(results: list[dict]):
    """Pretty-print retrieval results to the terminal."""
    print(f"\n{'='*60}")
    print(f"Top {len(results)} Results:")
    print(f"{'='*60}")
    for r in results:
        print(f"\n[Rank {r['rank']}] Score: {r['score']:.4f}")
        print(f"  Figure:  {r['figure_id']}")
        print(f"  arXiv:   {r['arxiv_id']}")
        print(f"  Caption: {r['caption'][:120]}...")
        print(f"  Image:   {r['image_path']}")


def main(args):
    retriever = SciCLIPRetriever(
        adapter_path=args.adapter_path,
        index_path=args.index_path,
        base_model=args.base_model,
    )

    if args.interactive:
        print("\nSciCLIP Interactive Search (type 'quit' to exit)")
        print("-" * 50)
        while True:
            query = input("\nQuery: ").strip()
            if query.lower() in ("quit", "exit", "q"):
                break
            if not query:
                continue
            results = retriever.search(query, top_k=args.top_k)
            print_results(results)
    else:
        print(f"\nQuery: '{args.query}'")
        results = retriever.search(args.query, top_k=args.top_k)
        print_results(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default="attention mechanism architecture")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--adapter_path", type=str,
                        default="checkpoints/lora_r8/best_adapter")
    parser.add_argument("--index_path", type=str, default="index/sciclip.faiss")
    parser.add_argument("--base_model", type=str,
                        default="openai/clip-vit-base-patch32")
    parser.add_argument("--interactive", action="store_true",
                        help="Enable interactive search mode")
    args = parser.parse_args()

    main(args)
