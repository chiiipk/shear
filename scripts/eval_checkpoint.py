#!/usr/bin/env python3
"""
Evaluate BGE-M3 model directly from .pt checkpoint files
Supports both Composer checkpoints and raw state_dict files
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset
from scipy.stats import spearmanr, pearsonr
import torch.nn.functional as F
from tqdm import tqdm
import os
import sys
import yaml
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.composer_bge_m3 import ComposerBGEM3
from omegaconf import DictConfig, OmegaConf


def create_model_config():
    """Create model configuration for loading checkpoint"""
    # Load from config_clean.yaml to match training setup
    config_path = Path(__file__).parent.parent / "config_clean.yaml"
    
    if config_path.exists():
        print(f"✅ Loading config from {config_path}")
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        # Convert to DictConfig and add missing defaults
        config = OmegaConf.create(yaml_config)
        
        # Add model architecture defaults if not present
        if 'model' not in config:
            config.model = {}
        
        # Set BGE-M3 defaults
        config.model.setdefault('base_model', 'BAAI/bge-m3')
        config.model.setdefault('d_model', 1024)
        config.model.setdefault('n_heads', 16)
        config.model.setdefault('n_layers', 24)
        config.model.setdefault('intermediate_size', 4096)
        config.model.setdefault('vocab_size', 250002)
        config.model.setdefault('max_position_embeddings', 8192)
        config.model.setdefault('use_sts_loss', True)
        config.model.setdefault('use_contrastive_loss', True)
        config.model.setdefault('temperature', 0.07)
        
        # Ensure l0_module exists
        if 'l0_module' not in config.model:
            config.model.l0_module = {}
        
        config.model.l0_module.setdefault('pruning_modules', ['layer', 'head', 'intermediate'])
        config.model.l0_module.setdefault('start_sparsity', 0.0)
        config.model.l0_module.setdefault('lagrangian_warmup_steps', '1000ba')
        config.model.l0_module.setdefault('eval_target_model', True)
        
        return config.model
    else:
        print("⚠️ config_clean.yaml not found, using defaults")
        # Fallback to hardcoded defaults
        config = DictConfig({
            'base_model': 'BAAI/bge-m3',
            'd_model': 1024,
            'n_heads': 16,
            'n_layers': 24,
            'intermediate_size': 4096,
            'vocab_size': 250002,
            'max_position_embeddings': 8192,
            'use_sts_loss': True,
            'use_contrastive_loss': True,
            'temperature': 0.07,
            'l0_module': {
                'pruning_modules': ['layer', 'head', 'intermediate'],
                'start_sparsity': 0.0,
                'lagrangian_warmup_steps': '1000ba',
                'eval_target_model': True,
            }
        })
        return config


def load_checkpoint_model(checkpoint_path: str):
    """Load ComposerBGEM3 model from checkpoint"""
    print(f"Loading checkpoint from {checkpoint_path}")
    
    # Load checkpoint - fix PyTorch 2.6+ pickle issue with OmegaConf
    try:
        # Try weights_only=True first (safer)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    except Exception:
        # Fallback to weights_only=False for OmegaConf compatibility
        print("⚠️ Using weights_only=False for OmegaConf compatibility")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract state dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print("✅ Loaded Composer checkpoint format")
    else:
        state_dict = checkpoint
        print("✅ Loaded raw state_dict format")
    
    # Create model configuration
    config = create_model_config()
    
    # Initialize model
    print("🔧 Initializing ComposerBGEM3 model...")
    model = ComposerBGEM3(config)
    
    # Check if this looks like an optimizer checkpoint instead of model weights
    model_keys = [k for k in state_dict.keys() if not k.startswith(('state', 'rng', 'param_groups'))]
    if len(model_keys) == 0:
        print("❌ This appears to be an optimizer checkpoint, not model weights")
        print("   Expected keys like 'backbone.*', 'l0_module.*', 'embedding_heads.*'")
        print("   Found keys:", list(state_dict.keys())[:10])
        return None, None
    
    # Load state dict
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        print(f"⚠️ Missing keys: {len(missing_keys)}")
        for key in missing_keys[:5]:  # Show first 5
            print(f"   - {key}")
        if len(missing_keys) > 5:
            print(f"   ... and {len(missing_keys) - 5} more")
    
    if unexpected_keys:
        print(f"⚠️ Unexpected keys: {len(unexpected_keys)}")
        for key in unexpected_keys[:5]:  # Show first 5
            print(f"   - {key}")
        if len(unexpected_keys) > 5:
            print(f"   ... and {len(unexpected_keys) - 5} more")
    
    # Move model to GPU if available and ensure all components are on same device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"📱 Model moved to: {device}")
    
    # Set to eval mode
    model.eval()
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Model loaded: {total_params:,} parameters")
    
    # Print pruning info if available
    if hasattr(model, 'l0_module'):
        try:
            zs = model.l0_module()
            print(f"🎯 Current pruning state:")
            for mask_name, mask_tensor in zs.items():
                sparsity = (mask_tensor == 0).float().mean().item()
                remaining = 1 - sparsity
                print(f"   {mask_name.replace('_z', '')}: {remaining:.1%} remaining ({sparsity:.1%} pruned)")
        except Exception as e:
            print(f"⚠️ Could not get pruning state: {e}")
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-m3')
        print("✅ Loaded BGE-M3 tokenizer")
    except Exception as e:
        print(f"❌ Could not load tokenizer: {e}")
        return None, None
    
    return model, tokenizer


def encode_texts(texts, model, tokenizer, batch_size=32, max_length=512):
    """Encode texts using the complete ComposerBGEM3 model"""
    model.eval()
    embeddings = []
    
    # Get model device
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            
            # Move inputs to model device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Forward through complete model (backbone + embedding heads)
            outputs = model(inputs)
            
            # Extract dense embeddings (BGE-M3 style)
            if 'embeddings' in outputs and 'dense_embedding' in outputs['embeddings']:
                # Use the proper BGE-M3 dense embeddings with learned transformations
                batch_embeddings = outputs['embeddings']['dense_embedding']
                print("✅ Using learned BGE-M3 dense embeddings")
            else:
                # Fallback: extract from backbone and use embedding heads manually
                backbone_outputs = model.backbone(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    return_dict=True
                )
                
                # Use embedding heads for proper BGE-M3 embeddings
                embedding_outputs = model.embedding_heads(
                    hidden_states=backbone_outputs.last_hidden_state,
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    return_dense=True,
                    return_sparse=False,
                    return_multi_vector=False,
                )
                
                batch_embeddings = embedding_outputs['dense_embedding']
                print("✅ Using BGE-M3 embedding heads")
            
            embeddings.append(batch_embeddings.cpu())
    
    return torch.cat(embeddings, dim=0)


def evaluate_sts(model, tokenizer, dataset_name="mteb/stsbenchmark-sts"):
    """Evaluate on STS dataset"""
    print(f"Loading STS dataset: {dataset_name}")
    
    # Load dataset - try MTEB format first, fallback to sentence-transformers
    try:
        dataset = load_dataset(dataset_name)
        print("✅ Using MTEB format (consistent with training)")
    except:
        print("⚠️ MTEB dataset not available, falling back to sentence-transformers format")
        dataset = load_dataset("sentence-transformers/stsb")
    
    test_data = dataset['test']
    
    # Extract sentences and scores
    sentences1 = test_data['sentence1']
    sentences2 = test_data['sentence2'] 
    scores = np.array(test_data['score'])
    
    print(f"Evaluating on {len(sentences1)} sentence pairs")
    
    # Encode sentences
    print("Encoding sentence 1...")
    embeddings1 = encode_texts(sentences1, model, tokenizer)
    
    print("Encoding sentence 2...")
    embeddings2 = encode_texts(sentences2, model, tokenizer)
    
    # Compute similarities
    similarities = F.cosine_similarity(embeddings1, embeddings2, dim=1)
    
    # Scale to [0, 5] range to match STS scores
    predicted_scores = (similarities + 1) * 2.5
    predicted_scores = predicted_scores.numpy()
    
    # Compute correlations
    spearman_corr, _ = spearmanr(predicted_scores, scores)
    pearson_corr, _ = pearsonr(predicted_scores, scores)
    
    print(f"\n📊 STS Evaluation Results:")
    print(f"Spearman correlation: {spearman_corr:.4f}")
    print(f"Pearson correlation: {pearson_corr:.4f}")
    
    return {
        'spearman': spearman_corr,
        'pearson': pearson_corr,
        'predicted_scores': predicted_scores,
        'true_scores': scores
    }


def analyze_checkpoint_info(checkpoint_path: str):
    """Analyze checkpoint metadata without loading the full model"""
    print(f"📋 Analyzing checkpoint: {checkpoint_path}")
    
    # Fix PyTorch 2.6+ pickle issue with OmegaConf
    try:
        # Try weights_only=True first (safer)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    except Exception:
        # Fallback to weights_only=False for OmegaConf compatibility
        print("⚠️ Using weights_only=False for OmegaConf compatibility")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'state_dict' in checkpoint:
        print("📦 Checkpoint type: Composer format")
        state_dict = checkpoint['state_dict']
        
        # Print training metadata if available
        if 'timestamp' in checkpoint:
            timestamp = checkpoint['timestamp']
            print(f"⏰ Training progress: {timestamp}")
        
        if 'run_name' in checkpoint:
            print(f"🏷️ Run name: {checkpoint['run_name']}")
            
        # Analyze optimizer state
        if 'optimizers' in checkpoint:
            print(f"🔧 Optimizer state available: {len(checkpoint['optimizers'])} optimizers")
    else:
        print("📦 Checkpoint type: Raw state_dict")
        state_dict = checkpoint
    
    # Analyze model components
    backbone_keys = [k for k in state_dict.keys() if k.startswith('backbone.')]
    l0_keys = [k for k in state_dict.keys() if k.startswith('l0_module.')]
    embedding_keys = [k for k in state_dict.keys() if k.startswith('embedding_heads.')]
    
    print(f"🧠 Model components:")
    print(f"   Backbone parameters: {len(backbone_keys)}")
    print(f"   L0 pruning parameters: {len(l0_keys)}")
    print(f"   Embedding head parameters: {len(embedding_keys)}")
    
    # Analyze L0 masks if available
    mask_keys = [k for k in l0_keys if 'masks.' in k and 'z_loga' in k]
    if mask_keys:
        print(f"🎯 Pruning masks found:")
        for key in mask_keys:
            mask_type = key.split('.')[-2]  # Extract mask type
            mask_tensor = state_dict[key]
            print(f"   {mask_type}: shape {mask_tensor.shape}")
    
    print()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate BGE-M3 model from checkpoint')
    parser.add_argument('checkpoint_path', type=str, 
                       help='Path to model checkpoint (.pt file)')
    parser.add_argument('--dataset', type=str, default='mteb/stsbenchmark-sts',
                       help='Dataset to evaluate on')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for encoding')
    parser.add_argument('--analyze_only', action='store_true',
                       help='Only analyze checkpoint info without evaluation')
    
    args = parser.parse_args()
    
    print("🧪 BGE-M3 Checkpoint Evaluation")
    print("=" * 50)
    
    # Analyze checkpoint info
    analyze_checkpoint_info(args.checkpoint_path)
    
    if args.analyze_only:
        print("✅ Checkpoint analysis complete!")
        return
    
    # Load model from checkpoint
    model, tokenizer = load_checkpoint_model(args.checkpoint_path)
    
    if model is None or tokenizer is None:
        print("❌ Failed to load model or tokenizer")
        return
    
    # Evaluate on STS
    print(f"\n🔍 Evaluating on {args.dataset}...")
    results = evaluate_sts(model, tokenizer, args.dataset)
    
    print(f"\n✅ Evaluation complete!")
    print(f"📊 Final Results:")
    print(f"   Spearman correlation: {results['spearman']:.4f}")
    print(f"   Pearson correlation: {results['pearson']:.4f}")
    
    # Performance interpretation
    spearman = results['spearman']
    if spearman >= 0.85:
        print("🎉 Excellent performance!")
    elif spearman >= 0.75:
        print("✅ Good performance")
    elif spearman >= 0.65:
        print("⚠️ Moderate performance")
    else:
        print("❌ Poor performance - check model and pruning state")


if __name__ == "__main__":
    main()
