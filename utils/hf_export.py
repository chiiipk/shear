"""Clean HuggingFace export utilities for BGE-M3 pruned models (compatible with masked backbone & 2D masks)"""

from __future__ import annotations
import os, json
from pathlib import Path
from typing import Dict, Optional
import math  
import sys   
import os   
import json 
import torch
from transformers import (
    AutoTokenizer,
    AutoConfig,
    XLMRobertaConfig,
    XLMRobertaModel,
)
from transformers.pytorch_utils import prune_linear_layer


# ----------------------------- small helpers -----------------------------

def _copy_embeddings(src_emb, dst_emb):
    dst_emb.load_state_dict(src_emb.state_dict())

def _assert_uniform(values, name: str):
    vals = list(values)
    if len(set(vals)) != 1:
        raise ValueError(
            f"[export_pruned_backbone_clean] non-uniform {name} across layers: {vals}. "
            f"Structural export requires uniform {name}. Use export_masked_backbone instead."
        )
    return vals[0]

def _device_dtype_like(t: torch.Tensor):
    return dict(device=t.device, dtype=t.dtype)


# ---------------------- config from (structurally) pruned ----------------------

def create_true_pruned_config(backbone) -> Dict:
    """
    Build a config reflecting the ACTUAL structurally pruned backbone.
    Assumes: uniform heads and intermediate size across layers.
    """
    num_layers = len(backbone.encoder.layer)

    hidden_size = backbone.embeddings.word_embeddings.weight.shape[1]
    vocab_size = backbone.embeddings.word_embeddings.weight.shape[0]
    max_pos    = backbone.embeddings.position_embeddings.weight.shape[0]
    type_vocab = backbone.embeddings.token_type_embeddings.weight.shape[0]

    # Validate uniformity across layers
    heads_per_layer = []
    inter_per_layer = []
    for lyr in backbone.encoder.layer:
        heads_per_layer.append(int(lyr.attention.num_attention_heads))
        inter_per_layer.append(int(lyr.intermediate.dense.out_features))

    num_heads = _assert_uniform(heads_per_layer, "num_attention_heads")
    intermediate_size = _assert_uniform(inter_per_layer, "intermediate_size")

    assert hidden_size % num_heads == 0, (
        f"Invalid config: hidden_size={hidden_size} not divisible by num_heads={num_heads}"
    )

    return {
        "architectures": ["XLMRobertaModel"],
        "attention_probs_dropout_prob": getattr(backbone.config, "attention_probs_dropout_prob", 0.1),
        "bos_token_id": 0,
        "eos_token_id": 2,
        "hidden_act": getattr(backbone.config, "hidden_act", "gelu"),
        "hidden_dropout_prob": getattr(backbone.config, "hidden_dropout_prob", 0.1),
        "hidden_size": hidden_size,
        "initializer_range": getattr(backbone.config, "initializer_range", 0.02),
        "intermediate_size": intermediate_size,
        "layer_norm_eps": getattr(backbone.config, "layer_norm_eps", 1e-5),
        "max_position_embeddings": max_pos,
        "model_type": "xlm-roberta",
        "num_attention_heads": num_heads,
        "num_hidden_layers": num_layers,
        "output_past": True,
        "pad_token_id": 1,
        "position_embedding_type": "absolute",
        "type_vocab_size": type_vocab,
        "use_cache": True,
        "vocab_size": vocab_size,
    }


# ---------------------- weight transfer (STRUCTURAL) ----------------------

def _prune_attention_modules_to_all_head_size(hf_self_attn, target_all_head_size: int):
    """
    Reduce Q/K/V out_features and attention.output.dense in_features to target_all_head_size
    by keeping the first indices (contiguous head block). Updates head_dim metadata too.
    """
    if target_all_head_size <= 0:
        raise ValueError("target_all_head_size must be > 0")

    dev = hf_self_attn.query.weight.device
    keep = torch.arange(target_all_head_size, device=dev)

    # prune rows (out_features) of q/k/v
    hf_self_attn.query = prune_linear_layer(hf_self_attn.query, keep, dim=0)
    hf_self_attn.key   = prune_linear_layer(hf_self_attn.key,   keep, dim=0)
    hf_self_attn.value = prune_linear_layer(hf_self_attn.value, keep, dim=0)

    # fix metadata
    # num_attention_heads should already be set by config; recompute head_dim/all_head_size for safety
    num_heads = hf_self_attn.num_attention_heads
    hf_self_attn.all_head_size = int(target_all_head_size)
    hf_self_attn.head_dim = hf_self_attn.all_head_size // num_heads

def transfer_pruned_weights(backbone, hf_model):
    """Copy structurally pruned weights from masked backbone -> fresh HF model.

    Changes (critical fix):
      * Copy the actually LEARNED attention output projection (self_output.dense)
        instead of fabricating an identity mapping. The previous identity
        initialization caused a severe train/export mismatch and representation loss.
    """
    # embeddings
    _copy_embeddings(backbone.embeddings, hf_model.embeddings)

    # layers
    for i, b_layer in enumerate(backbone.encoder.layer):
        h_layer = hf_model.encoder.layer[i]

        # dimensions
        hidden_size = backbone.config.hidden_size
        all_head_size = int(b_layer.attention.all_head_size)  # pruned rows in Q/K/V
        num_heads = int(b_layer.attention.num_attention_heads)

        # ---- attention ----
        # prune HF attention to target all_head_size THEN copy weights
        _prune_attention_modules_to_all_head_size(h_layer.attention.self, all_head_size)
        h_self = h_layer.attention.self

        # copy Q/K/V (now matching shapes)
        h_self.query.weight.data.copy_(b_layer.attention.query.weight.data)
        h_self.query.bias.data.copy_(b_layer.attention.query.bias.data)
        h_self.key.weight.data.copy_(b_layer.attention.key.weight.data)
        h_self.key.bias.data.copy_(b_layer.attention.key.bias.data)
        h_self.value.weight.data.copy_(b_layer.attention.value.weight.data)
        h_self.value.bias.data.copy_(b_layer.attention.value.bias.data)

        # prune input of attention.output.dense to all_head_size (keep first slice)
        # then copy learned projection weights from backbone self_output.dense
        dev = h_layer.attention.output.dense.weight.device
        keep_cols = torch.arange(all_head_size, device=dev)
        h_layer.attention.output.dense = prune_linear_layer(
            h_layer.attention.output.dense, keep_cols, dim=1
        )

        # Copy learned projection (shape: [hidden_size, all_head_size])
        learned_out = b_layer.self_output.dense
        h_out = h_layer.attention.output.dense
        if h_out.weight.shape != learned_out.weight.shape:
            raise ValueError(
                f"Projection shape mismatch after pruning: HF {h_out.weight.shape} vs backbone {learned_out.weight.shape}"
            )
        h_out.weight.data.copy_(learned_out.weight.data)
        h_out.bias.data.copy_(learned_out.bias.data)

        # ---- MLP ----
        h_layer.intermediate.dense.weight.data.copy_(b_layer.intermediate.dense.weight.data)
        h_layer.intermediate.dense.bias.data.copy_(b_layer.intermediate.dense.bias.data)
        h_layer.output.dense.weight.data.copy_(b_layer.output.dense.weight.data)
        h_layer.output.dense.bias.data.copy_(b_layer.output.dense.bias.data)

        # ---- LayerNorms ----
        # backbone uses single LN after attn+FF; mirror it into both HF LNs
        h_layer.attention.output.LayerNorm.weight.data.copy_(b_layer.output.LayerNorm.weight.data)
        h_layer.attention.output.LayerNorm.bias.data.copy_(b_layer.output.LayerNorm.bias.data)
        h_layer.output.LayerNorm.weight.data.copy_(b_layer.output.LayerNorm.weight.data)
        h_layer.output.LayerNorm.bias.data.copy_(b_layer.output.LayerNorm.bias.data)

        # fix metadata just in case
        h_self.num_attention_heads = num_heads
        h_self.all_head_size = all_head_size
        h_self.head_dim = all_head_size // num_heads


# ----------------------- weight copy (MASKED / original dims) -----------------------

def _copy_layer_weights_no_structural_change(b_layer, h_layer, hidden_size: int):
    """Name-safe copy for original-dim (masked) export.

    Critical change: copy the learned attention output projection instead of
    forcing an identity matrix. This preserves the trained representation.
    """
    # Q/K/V
    h_layer.attention.self.query.weight.data.copy_(b_layer.attention.query.weight.data)
    h_layer.attention.self.query.bias.data.copy_(b_layer.attention.query.bias.data)
    h_layer.attention.self.key.weight.data.copy_(b_layer.attention.key.weight.data)
    h_layer.attention.self.key.bias.data.copy_(b_layer.attention.key.bias.data)
    h_layer.attention.self.value.weight.data.copy_(b_layer.attention.value.weight.data)
    h_layer.attention.self.value.bias.data.copy_(b_layer.attention.value.bias.data)

    # learned out-proj
    h_out = h_layer.attention.output.dense
    learned_out = b_layer.self_output.dense
    h_out.weight.data.copy_(learned_out.weight.data)
    h_out.bias.data.copy_(learned_out.bias.data)

    # MLP
    h_layer.intermediate.dense.weight.data.copy_(b_layer.intermediate.dense.weight.data)
    h_layer.intermediate.dense.bias.data.copy_(b_layer.intermediate.dense.bias.data)
    h_layer.output.dense.weight.data.copy_(b_layer.output.dense.weight.data)
    h_layer.output.dense.bias.data.copy_(b_layer.output.dense.bias.data)

    # LayerNorms
    h_layer.attention.output.LayerNorm.weight.data.copy_(b_layer.output.LayerNorm.weight.data)
    h_layer.attention.output.LayerNorm.bias.data.copy_(b_layer.output.LayerNorm.bias.data)
    h_layer.output.LayerNorm.weight.data.copy_(b_layer.output.LayerNorm.weight.data)
    h_layer.output.LayerNorm.bias.data.copy_(b_layer.output.LayerNorm.bias.data)


def apply_learned_masks(backbone, hf_model, zs: Dict[str, torch.Tensor]):
    """
    Apply 2-D masks (layer_z:[L], head_z:[L,H], intermediate_z:[L,I]) onto an HF model with original dims.
    """
    _copy_embeddings(backbone.embeddings, hf_model.embeddings)

    L = len(backbone.encoder.layer)
    hidden_size = backbone.config.hidden_size
    H = hf_model.config.num_attention_heads
    I = hf_model.config.intermediate_size

    layer_mask = zs.get('layer_z', torch.ones(L, device=hf_model.device))
    head_mask = zs.get('head_z', torch.ones(L, H, device=hf_model.device))
    int_mask  = zs.get('intermediate_z', torch.ones(L, I, device=hf_model.device))

    # Squeeze potential broadcast dimensions introduced by mask_output_shape
    if isinstance(head_mask, torch.Tensor) and head_mask.dim() == 4 and head_mask.shape[1] == 1 and head_mask.shape[-1] == 1:
        head_mask = head_mask.squeeze(1).squeeze(-1)  # [L,H]
    if isinstance(int_mask, torch.Tensor) and int_mask.dim() == 4 and int_mask.shape[1] == 1 and int_mask.shape[2] == 1:
        int_mask = int_mask.squeeze(1).squeeze(1)  # [L,I]
    if isinstance(layer_mask, torch.Tensor) and layer_mask.dim() > 1:
        layer_mask = layer_mask.view(layer_mask.shape[0])  # flatten extras

    for i in range(min(L, len(hf_model.encoder.layer))):
        h_layer = hf_model.encoder.layer[i]
        b_layer = backbone.encoder.layer[i]

        if i < layer_mask.shape[0]:
            lm_val = layer_mask[i]
            # Ensure scalar
            if lm_val.numel() > 1:
                lm_val = lm_val.view(-1)[0]
            layer_active = (lm_val.item() > 0)
        else:
            layer_active = True
        if not layer_active:
            for p in h_layer.parameters():
                p.data.zero_()
            continue

        # copy weights first
        _copy_layer_weights_no_structural_change(b_layer, h_layer, hidden_size)

        # head masking (zero contiguous head spans)
        if isinstance(head_mask, torch.Tensor) and i < head_mask.shape[0]:
            layer_head_mask = head_mask[i].view(-1)
            head_dim = hidden_size // H
            for head_idx in range(min(H, layer_head_mask.shape[0])):
                if layer_head_mask[head_idx].item() == 0:
                    s, e = head_idx * head_dim, (head_idx + 1) * head_dim
                    for name in ("query", "key", "value"):
                        lin = getattr(h_layer.attention.self, name)
                        lin.weight.data[s:e, :] = 0
                        lin.bias.data[s:e] = 0

        # intermediate masking
        if isinstance(int_mask, torch.Tensor) and i < int_mask.shape[0]:
            layer_int_mask = int_mask[i].view(-1)
            for j in range(min(I, layer_int_mask.shape[0])):
                if layer_int_mask[j].item() == 0:
                    h_layer.intermediate.dense.weight.data[j, :] = 0
                    h_layer.intermediate.dense.bias.data[j] = 0
                    h_layer.output.dense.weight.data[:, j] = 0


# ----------------------------- export entry points -----------------------------

def export_embedding_heads(embedding_heads, save_path: str):
    """Export embedding heads for completeness (dense/sparse/multi-vector)."""
    heads_path = os.path.join(save_path, "embedding_heads.pt")
    heads_state = {
        "dense_head": getattr(embedding_heads, "dense_head", None).state_dict() if hasattr(embedding_heads, "dense_head") else None,
        "sparse_head": getattr(embedding_heads, "sparse_head", None).state_dict() if hasattr(embedding_heads, "sparse_head") else None,
        "multi_vector_head": getattr(embedding_heads, "multi_vector_head", None).state_dict() if hasattr(embedding_heads, "multi_vector_head") else None,
        "vocab_size": getattr(embedding_heads, "vocab_size", None),
        "hidden_size": getattr(embedding_heads, "hidden_size", None),
    }
    torch.save(heads_state, heads_path)
    print(f"✅ Embedding heads saved to {heads_path}")

def export_pruned_backbone_clean(backbone, save_path: str, base_model_name: str = "BAAI/bge-m3", embedding_heads=None):
    """
    Export a structurally pruned HF model that mirrors the masked backbone shapes.
    Requires uniform heads & intermediate size across layers.
    """
    Path(save_path).mkdir(parents=True, exist_ok=True)

    pruned_cfg = create_true_pruned_config(backbone)  # raises if non-uniform
    hf = XLMRobertaModel(XLMRobertaConfig(**pruned_cfg))

    transfer_pruned_weights(backbone, hf)  # structural copy

    # save model + tokenizer
    hf.save_pretrained(save_path)
    tok = AutoTokenizer.from_pretrained(base_model_name)
    tok.save_pretrained(save_path)

    if embedding_heads is not None:
        export_embedding_heads(embedding_heads, save_path)
        with open(os.path.join(save_path, "model_config.json"), "w") as f:
            json.dump({
                "architecture": "bge-m3-pruned",
                "has_embedding_heads": True,
                "base_model": base_model_name,
                "head_config": {
                    "dense_dim": getattr(embedding_heads, "dense_head", None).hidden_size if hasattr(embedding_heads, "dense_head") else None,
                    "vocab_size": getattr(embedding_heads, "vocab_size", None),
                    "hidden_size": getattr(embedding_heads, "hidden_size", None),
                },
            }, f, indent=2)

    print(f"✅ Clean pruned model exported to {save_path}")
    print(f"📊 Architecture: {pruned_cfg['num_hidden_layers']} layers, "
          f"{pruned_cfg['num_attention_heads']} heads, {pruned_cfg['intermediate_size']} intermediate")

    return save_path


# Backward-compat wrappers
def save_backbone_as_hf_model(backbone, save_path, base_model_name="BAAI/bge-m3"):
    return export_pruned_backbone_clean(backbone, save_path, base_model_name)

def create_hf_config_from_backbone(backbone):
    return create_true_pruned_config(backbone)


# ---------------------- masked export (original dimensions) ----------------------

def create_original_config(base_model_name: str = "BAAI/bge-m3") -> Dict:
    return AutoConfig.from_pretrained(base_model_name).to_dict()

def export_masked_backbone(backbone, save_path: str, base_model_name: str = "BAAI/bge-m3",
                           embedding_heads=None, zs: Optional[Dict[str, torch.Tensor]] = None):
    """
    Export model with ORIGINAL dimensions; optionally applies learned masks (zeros).
    Safe when heads/intermediate vary per layer.
    """
    Path(save_path).mkdir(parents=True, exist_ok=True)

    orig_cfg = create_original_config(base_model_name)
    hf = XLMRobertaModel(XLMRobertaConfig(**orig_cfg))

    hidden_size = backbone.config.hidden_size

    _copy_embeddings(backbone.embeddings, hf.embeddings)
    L = min(len(backbone.encoder.layer), len(hf.encoder.layer))
    for i in range(L):
        _copy_layer_weights_no_structural_change(backbone.encoder.layer[i], hf.encoder.layer[i], hidden_size)

    if zs is not None:
        apply_learned_masks(backbone, hf, zs)

    hf.save_pretrained(save_path)
    tok = AutoTokenizer.from_pretrained(base_model_name)
    tok.save_pretrained(save_path)

    if embedding_heads is not None:
        export_embedding_heads(embedding_heads, save_path)
        model_cfg = {
            "architecture": "bge-m3-masked",
            "export_type": "masked",
            "has_embedding_heads": True,
            "base_model": base_model_name,
            "original_dimensions": True,
            "head_config": {
                "dense_dim": getattr(embedding_heads, "dense_head", None).hidden_size if hasattr(embedding_heads, "dense_head") else None,
                "vocab_size": getattr(embedding_heads, "vocab_size", None),
                "hidden_size": getattr(embedding_heads, "hidden_size", None),
            },
        }
        if zs is not None:
            model_cfg["mask_info"] = {
                name: {"shape": list(t.shape), "sparsity": float((t == 0).float().mean())}
                for name, t in zs.items()
            }
        with open(os.path.join(save_path, "model_config.json"), "w") as f:
            json.dump(model_cfg, f, indent=2)

    print(f"✅ Masked model exported to {save_path}")
    print(f"📊 Architecture: {orig_cfg['num_hidden_layers']} layers, "
          f"{orig_cfg['num_attention_heads']} heads, {orig_cfg['intermediate_size']} intermediate (original dims)")
    if zs is not None:
        for name, t in zs.items():
            print(f"🎯 {name}: {(t == 0).float().mean().item():.1%} zeroed")

    return save_path
