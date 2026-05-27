"""
app.py

Gradio web demo for SciCLIP.
Provides a side-by-side comparison of Vanilla CLIP vs LoRA-CLIP retrieval.

Usage:
    python app.py
    python app.py --share    # public Gradio link
    python app.py --adapter_path checkpoints/lora_r8/best_adapter
"""

import argparse
import gradio as gr
from PIL import Image
from pathlib import Path

from retrieve import SciCLIPRetriever


# -------------------------------------------------------------------
# App configuration
# -------------------------------------------------------------------
BASE_MODEL = "openai/clip-vit-base-patch32"
DEFAULT_ADAPTER = "checkpoints/lora_r8/best_adapter"
DEFAULT_INDEX = "index/sciclip.faiss"
TOP_K = 5

EXAMPLE_QUERIES = [
    "attention mechanism transformer architecture diagram",
    "training loss curve comparison baseline",
    "BLEU score bar chart evaluation results",
    "neural network layers visualization",
    "confusion matrix classification results",
    "UMAP embedding visualization scatter plot",
    "ablation study table results",
    "encoder decoder sequence to sequence model",
]


def build_retrievers(adapter_path: str, index_path: str):
    """Initialize both baseline and LoRA-CLIP retrievers."""
    print("Initializing LoRA-CLIP retriever...")
    lora_retriever = SciCLIPRetriever(
        adapter_path=adapter_path,
        index_path=index_path,
        base_model=BASE_MODEL,
    )

    # For vanilla CLIP baseline, we use a dummy adapter path
    # In practice, build a separate index with vanilla CLIP embeddings
    # For demo, we show the same retriever (replace with vanilla index if available)
    baseline_retriever = None
    baseline_index = index_path.replace("sciclip.faiss", "sciclip_baseline.faiss")
    if Path(baseline_index).exists():
        print("Initializing vanilla CLIP baseline retriever...")
        baseline_retriever = SciCLIPRetriever(
            adapter_path=None,  # No LoRA adapter
            index_path=baseline_index,
            base_model=BASE_MODEL,
        )

    return lora_retriever, baseline_retriever


def search_and_format(retriever, query: str, top_k: int = TOP_K):
    """Run search and format results as (image, caption) pairs for Gallery."""
    if retriever is None:
        return [], "Baseline index not available. Run build_index.py first."

    results = retriever.search(query, top_k=top_k)
    gallery_items = []

    for r in results:
        try:
            img = Image.open(r["image_path"]).convert("RGB")
            label = (
                f"#{r['rank']} | Score: {r['score']:.3f}\n"
                f"{r['caption'][:80]}..."
            )
            gallery_items.append((img, label))
        except Exception:
            pass

    return gallery_items


def create_demo(lora_retriever, baseline_retriever):
    """Build the Gradio interface."""

    with gr.Blocks(
        title="SciCLIP — Scientific Figure Retrieval",
        theme=gr.themes.Soft(primary_hue="indigo"),
        css="""
        .header { text-align: center; margin-bottom: 20px; }
        .subtitle { color: #666; font-size: 0.9em; text-align: center; }
        """,
    ) as demo:

        # Header
        gr.HTML("""
        <div class="header">
            <h1>🔬 SciCLIP</h1>
            <h3>Scientific Figure Retrieval with LoRA-Adapted CLIP</h3>
            <p class="subtitle">
                Type a description of a figure you're looking for.
                SciCLIP retrieves the most relevant figures from arXiv papers.
            </p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=3):
                query_input = gr.Textbox(
                    label="Search Query",
                    placeholder="e.g. attention mechanism transformer architecture",
                    lines=2,
                )
            with gr.Column(scale=1):
                top_k_slider = gr.Slider(
                    minimum=1, maximum=10, value=5, step=1,
                    label="Number of results",
                )
                search_btn = gr.Button("🔍 Search", variant="primary")

        # Example queries
        gr.Examples(
            examples=EXAMPLE_QUERIES,
            inputs=query_input,
            label="Example Queries",
        )

        # Results: side-by-side comparison
        with gr.Tabs():
            with gr.TabItem("🧠 LoRA-CLIP (Fine-tuned)"):
                lora_gallery = gr.Gallery(
                    label="LoRA-CLIP Results",
                    columns=5,
                    height="auto",
                    object_fit="contain",
                )
                lora_status = gr.Markdown("")

            if baseline_retriever is not None:
                with gr.TabItem("📊 Vanilla CLIP (Baseline)"):
                    baseline_gallery = gr.Gallery(
                        label="Vanilla CLIP Results",
                        columns=5,
                        height="auto",
                        object_fit="contain",
                    )
                    baseline_status = gr.Markdown("")

        # About section
        gr.Markdown("""
        ---
        ### About SciCLIP
        **SciCLIP** fine-tunes [CLIP](https://openai.com/research/clip) with
        [LoRA](https://arxiv.org/abs/2106.09685) adapters on the
        [SciCap](https://huggingface.co/datasets/vector-institute/SciCap) dataset
        to improve retrieval of scientific figures from arXiv papers.

        **Model**: `openai/clip-vit-base-patch32` + LoRA (r=8)  
        **Dataset**: SciCap (~20k figure-caption pairs from cs.CL, cs.LG, cs.CV)  
        **Metrics**: R@1, R@5, R@10, MRR (text→image retrieval)
        """)

        # Event handlers
        def on_search(query, top_k):
            if not query.strip():
                return [], "Please enter a search query."

            lora_results = search_and_format(lora_retriever, query, top_k)
            status = f"Found {len(lora_results)} results for: **{query}**"
            return lora_results, status

        search_btn.click(
            fn=on_search,
            inputs=[query_input, top_k_slider],
            outputs=[lora_gallery, lora_status],
        )
        query_input.submit(
            fn=on_search,
            inputs=[query_input, top_k_slider],
            outputs=[lora_gallery, lora_status],
        )

    return demo


def main(args):
    lora_retriever, baseline_retriever = build_retrievers(
        adapter_path=args.adapter_path,
        index_path=args.index_path,
    )
    demo = create_demo(lora_retriever, baseline_retriever)
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", type=str, default=DEFAULT_ADAPTER)
    parser.add_argument("--index_path", type=str, default=DEFAULT_INDEX)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true",
                        help="Create public Gradio link")
    args = parser.parse_args()
    main(args)
