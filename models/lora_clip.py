"""
models/lora_clip.py

LoRA-adapted CLIP model for scientific figure retrieval.
Uses HuggingFace PEFT to inject low-rank adapters into CLIP's
attention layers (Q, K, V projections) in both visual and text encoders.

Usage:
    model = LoRACLIP()
    img_emb = model.encode_image(pixel_values)   # (B, 512)
    txt_emb = model.encode_text(input_ids, attention_mask)  # (B, 512)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from peft import LoraConfig, get_peft_model, TaskType


# -------------------------------------------------------------------
# Default LoRA configuration
# -------------------------------------------------------------------
DEFAULT_LORA_CONFIG = LoraConfig(
    # Rank of the low-rank decomposition. Higher = more expressive but more params.
    # Ablation values: 4, 8, 16
    r=8,
    # Scaling factor: alpha / r controls effective learning rate of adapters
    lora_alpha=16,
    # PEFT will find all layers named q_proj/k_proj/v_proj in both encoders.
    # Note: do NOT use wildcard paths like 'encoder.layers.*.self_attn.q_proj'
    # as they are not supported in PEFT >= 0.9. Simple names work across versions.
    target_modules=["q_proj", "k_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    # PEFT doesn't have a CLIP task type; we use FEATURE_EXTRACTION
    task_type=TaskType.FEATURE_EXTRACTION,
)


class LoRACLIP(nn.Module):
    """
    CLIP model with LoRA adapters injected into attention layers.

    Architecture:
        Base: openai/clip-vit-base-patch32
        LoRA: Applied to Q, K, V projections in visual + text encoders
        Embedding dim: 512

    Parameters:
        model_name: HuggingFace CLIP model identifier
        lora_config: PEFT LoraConfig. If None, uses DEFAULT_LORA_CONFIG
        freeze_base: Whether to freeze all base model params (LoRA trains separately)
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        lora_config: LoraConfig = None,
        freeze_base: bool = True,
    ):
        super().__init__()

        # Load base CLIP model and processor
        self.clip = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.embed_dim = self.clip.config.projection_dim  # 512 for ViT-B/32

        # Apply LoRA via PEFT
        config = lora_config if lora_config is not None else DEFAULT_LORA_CONFIG
        self.clip = get_peft_model(self.clip, config)

        if freeze_base:
            # Freeze all parameters, then unfreeze only LoRA adapter weights
            for name, param in self.clip.named_parameters():
                if "lora_" not in name:
                    param.requires_grad = False

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Encode images into normalized embedding vectors.

        Args:
            pixel_values: (B, 3, 224, 224) preprocessed image tensor

        Returns:
            image_embeds: (B, 512) L2-normalized image embeddings
        """
        # Get vision features and project to joint embedding space
        vision_outputs = self.clip.vision_model(pixel_values=pixel_values)
        pooled = vision_outputs.pooler_output          # (B, hidden_dim)
        image_embeds = self.clip.visual_projection(pooled)  # (B, 512)
        return F.normalize(image_embeds, dim=-1)

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode text queries into normalized embedding vectors.

        Args:
            input_ids: (B, seq_len) tokenized text
            attention_mask: (B, seq_len) attention mask

        Returns:
            text_embeds: (B, 512) L2-normalized text embeddings
        """
        text_outputs = self.clip.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # CLS token pooling (index of EOS token in CLIP)
        pooled = text_outputs.pooler_output               # (B, hidden_dim)
        text_embeds = self.clip.text_projection(pooled)   # (B, 512)
        return F.normalize(text_embeds, dim=-1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for training. Returns embeddings and learned temperature.

        Returns:
            image_embeds: (B, 512)
            text_embeds:  (B, 512)
            logit_scale:  scalar (log temperature parameter)
        """
        image_embeds = self.encode_image(pixel_values)
        text_embeds = self.encode_text(input_ids, attention_mask)
        logit_scale = self.clip.logit_scale.exp()
        return image_embeds, text_embeds, logit_scale

    def print_trainable_parameters(self):
        """Print LoRA adapter parameter count vs total parameters."""
        self.clip.print_trainable_parameters()

    def save_lora_adapter(self, save_path: str):
        """Save only the LoRA adapter weights (lightweight, ~MB)."""
        self.clip.save_pretrained(save_path)
        print(f"LoRA adapter saved to: {save_path}")

    @classmethod
    def from_pretrained_lora(
        cls,
        base_model_name: str,
        lora_adapter_path: str,
    ) -> "LoRACLIP":
        """Load a LoRA-CLIP model from a saved adapter checkpoint."""
        from peft import PeftModel

        instance = cls.__new__(cls)
        nn.Module.__init__(instance)

        base_clip = CLIPModel.from_pretrained(base_model_name)
        instance.clip = PeftModel.from_pretrained(base_clip, lora_adapter_path)
        instance.processor = CLIPProcessor.from_pretrained(base_model_name)
        instance.embed_dim = base_clip.config.projection_dim
        return instance


# -------------------------------------------------------------------
# InfoNCE (Symmetric Cross-Entropy) Contrastive Loss
# -------------------------------------------------------------------
def contrastive_loss(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """
    Symmetric InfoNCE loss (same as original CLIP training objective).

    For a batch of B (image, text) pairs, the loss encourages:
      - High similarity between matched pairs (diagonal)
      - Low similarity between unmatched pairs (off-diagonal)

    Args:
        image_embeds: (B, D) normalized image embeddings
        text_embeds:  (B, D) normalized text embeddings
        logit_scale:  scalar temperature (learned, initialized to 1/0.07)

    Returns:
        loss: scalar contrastive loss
    """
    batch_size = image_embeds.shape[0]

    # Compute similarity matrix: (B, B)
    logits_per_image = logit_scale * image_embeds @ text_embeds.T
    logits_per_text = logit_scale * text_embeds @ image_embeds.T

    # Ground truth: each image matches its own text (diagonal)
    labels = torch.arange(batch_size, device=image_embeds.device)

    # Symmetric cross-entropy
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)

    return (loss_i + loss_t) / 2.0


if __name__ == "__main__":
    # Quick sanity check
    print("Loading LoRA-CLIP model...")
    model = LoRACLIP(lora_config=DEFAULT_LORA_CONFIG)
    model.print_trainable_parameters()

    # Dummy forward pass
    dummy_pixels = torch.randn(4, 3, 224, 224)
    dummy_ids = torch.randint(0, 49408, (4, 77))
    dummy_mask = torch.ones(4, 77, dtype=torch.long)

    img_emb, txt_emb, scale = model(dummy_pixels, dummy_ids, dummy_mask)
    loss = contrastive_loss(img_emb, txt_emb, scale)

    print(f"Image embeddings: {img_emb.shape}")   # (4, 512)
    print(f"Text embeddings:  {txt_emb.shape}")   # (4, 512)
    print(f"Contrastive loss: {loss.item():.4f}")
    print("✓ Sanity check passed!")
