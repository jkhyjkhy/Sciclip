# SciCLIP 🔬

**LoRA-Adapted CLIP for Scientific Figure Retrieval**

> Find figures from arXiv papers using natural language descriptions.  
> *"attention mechanism architecture diagram"* → retrieves relevant figures instantly.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

Standard CLIP was trained on natural images and struggles with the abstract visual language of scientific figures (graphs, architecture diagrams, loss curves, etc.).

**SciCLIP** addresses this by fine-tuning CLIP with [LoRA](https://arxiv.org/abs/2106.09685) adapters on the [SciCap](https://huggingface.co/datasets/vector-institute/SciCap) dataset — a collection of real arXiv figures with captions.

| | Vanilla CLIP | SciCLIP (LoRA r=8) |
|--|:--:|:--:|
| R@1 | 0.1090 | 0.1655 |
| R@5 | 0.1935 | 0.2960 |
| R@10 | 0.2380 | 0.3545 |
| MRR | 0.1564 | 0.2324 |
| Trainable params | 100% | **~0.5%** |

---

## Quick Start

```bash
# 1. Clone & install
git clone https://github.com/YOUR_USERNAME/sciclip.git
cd sciclip
pip install -r requirements.txt

# 2. Download & prepare SciCap data (~20k samples)
python data/prepare_scicap.py --max_samples 20000

# 3. Train LoRA-CLIP (~2 hrs on Colab A100)
python train.py --lora_r 8 --epochs 5

# 4. Build FAISS retrieval index
python build_index.py

# 5. Launch Gradio demo
python app.py
```

Open `http://localhost:7860` and start searching!

---

## Usage

### CLI Search
```bash
python retrieve.py --query "transformer attention mechanism diagram"
python retrieve.py --interactive   # interactive mode
```

### Evaluate (Baseline vs LoRA-CLIP)
```bash
python evaluate.py \
    --adapter_path checkpoints/lora_r8/best_adapter \
    --val_path data/scicap_val.jsonl
```

### Ablation Study (LoRA rank)
```bash
python train.py --ablation   # trains r=4, r=8, r=16 sequentially
```

---

## Architecture

```
Text Query
    ↓
LoRA-CLIP Text Encoder (CLIP ViT-B/32 + LoRA on Q,K,V)
    ↓
Query Embedding (512-dim, L2 normalized)
    ↓
FAISS IVFFlat Index Search
    ↓
Top-K Figures + Captions

[Offline]
SciCap Figures → LoRA-CLIP Vision Encoder → FAISS Index
```

**LoRA is applied to Q, K, V attention projections in both:**
- Visual encoder (12 transformer blocks)
- Text encoder (12 transformer blocks)

With `r=8`, only **~0.5% of parameters** are trainable vs full fine-tuning.

---

## Dataset

**SciCap** — Scientific Figure Caption dataset
- Source: Real figures from arXiv papers
- Available on HuggingFace: `vector-institute/SciCap`
- We use a filtered subset: cs.CL, cs.LG, cs.CV papers
- ~20k figure-caption pairs for training

---

## Project Structure

```
sciclip/
├── data/
│   └── prepare_scicap.py     # Download and preprocess SciCap
├── models/
│   └── lora_clip.py          # LoRA-CLIP model + InfoNCE loss
├── train.py                  # Training loop (contrastive learning)
├── evaluate.py               # Recall@k, MRR evaluation
├── build_index.py            # Build FAISS retrieval index
├── retrieve.py               # CLI search interface
├── app.py                    # Gradio demo UI
├── notebooks/
│   └── demo.ipynb            # Colab-ready notebook
└── requirements.txt
```

---

## Run on Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]([notebooks/demo.ipynb](https://drive.google.com/file/d/1rsmRPA37vreScRY4gcr-IQTs9jB4VF7R/view?usp=sharing))

Use Runtime → Change runtime type → **A100 GPU** for best performance.

---

## HuggingFace

Model is available on HF repo!

[![LoRA4]🥉](https://huggingface.co/jkhyjkhy/sciclip-lora-r4)  
[![LoRA8]🥇](https://huggingface.co/jkhyjkhy/sciclip-lora-r8)  
[![LoRA16]🥈](https://huggingface.co/jkhyjkhy/sciclip-lora-r16)  

---

## Citation

If you use this project, please cite:
```bibtex
@misc{sciclip2025,
  title   = {SciCLIP: LoRA-Adapted CLIP for Scientific Figure Retrieval},
  author  = {Younghee Jeong},
  year    = {2025},
  url     = {https://github.com/jkhyjkhy/sciclip}
}
```

**References:**
- Radford et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision.* (CLIP)
- Hu et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models.*
- Hsu et al. (2021). *SciCap: Generating Captions for Scientific Figures.*

---

## License

MIT
