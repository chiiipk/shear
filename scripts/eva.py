#!/usr/bin/env python3
"""
Evaluate pruned BGE-M3 model on STS dataset
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from scipy.stats import spearmanr, pearsonr
import torch.nn.functional as F
from tqdm import tqdm
import os


def load_embedding_head(model_path):
    """Load the embedding head from pruned model - handles both old and new formats"""
    heads_path = os.path.join(model_path, "embedding_heads.pt")
    
    if not os.path.exists(heads_path):
        print("⚠️ No embedding heads found, will use CLS token pooling")
        return None
    
    print(f"Loading embedding heads from {heads_path}")
    try:
        heads_data = torch.load(heads_path, map_location='cpu')
    except Exception as e:
        print(f"❌ Error loading embedding heads: {e}")
        print("   Will use CLS token pooling instead")
        return None
    
    # Check if we have the old format (with dense layer) or new format (simplified)
    try:
        old_format = 'dense_head' in heads_data and isinstance(heads_data['dense_head'], dict) and 'dense.weight' in heads_data['dense_head']
    except Exception:
        print("⚠️ Unrecognized embedding heads format, using CLS token pooling")
        return None
    
    if old_format:
        print("📦 Detected old embedding head format (with dense layer)")
        
        # Old format - recreate dense head with linear layer
        class DenseEmbeddingHeadOld(nn.Module):
            def __init__(self, hidden_size: int, output_dim: int = None):
                super().__init__()
                self.output_dim = output_dim or hidden_size
                self.dense = nn.Linear(hidden_size, self.output_dim)
                self.activation = nn.Tanh()
                
            def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
                # Use CLS token (BGE-M3 standard)
                pooled = hidden_states[:, 0]  # CLS token [batch_size, hidden_size]
                
                # Apply dense transformation
                dense_embedding = self.dense(pooled)
                dense_embedding = self.activation(dense_embedding)
                return F.normalize(dense_embedding, p=2, dim=-1)
        
        # Create and load the dense head
        hidden_size = heads_data['hidden_size']
        dense_head = DenseEmbeddingHeadOld(hidden_size)
        dense_head.load_state_dict(heads_data['dense_head'])
        print("✅ Loaded old format embedding head with learned transformations")
        return dense_head
    
    else:
        print("📦 Detected new simplified embedding head format (CLS token only)")
        
        # New simplified format - just CLS token with normalization
        class DenseEmbeddingHeadSimple(nn.Module):
            def __init__(self, hidden_size: int):
                super().__init__()
                self.hidden_size = hidden_size
                
            def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
                # BGE-M3 style: Use CLS token directly with normalization
                cls_embedding = hidden_states[:, 0]  # CLS token [batch_size, hidden_size]
                return F.normalize(cls_embedding, p=2, dim=-1)
        
        hidden_size = heads_data.get('hidden_size', 1024)  # Default BGE-M3 size
        dense_head = DenseEmbeddingHeadSimple(hidden_size)
        print("✅ Using simplified CLS token embedding approach")
        return dense_head


def load_pruned_model(model_path):
    """Load the pruned model, tokenizer, and embedding head"""
    print(f"Loading pruned model from {model_path}")
    
    # Load backbone
    model = AutoModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load embedding head
    embedding_head = load_embedding_head(model_path)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    if embedding_head is not None:
        total_params += sum(p.numel() for p in embedding_head.parameters())
    
    print(f"Model loaded: {total_params:,} parameters")
    print(f"Layers: {model.config.num_hidden_layers}")
    print(f"Attention heads: {model.config.num_attention_heads}")
    print(f"Intermediate size: {model.config.intermediate_size}")
    print(f"Using embedding head: {embedding_head is not None}")
    
    return model, tokenizer, embedding_head


def encode_texts(texts, model, tokenizer, embedding_head=None, batch_size=32, max_length=512):
    """Encode texts using the model with proper embedding head"""
    model.eval()
    if embedding_head is not None:
        embedding_head.eval()
    
    embeddings = []
    
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
            
            # Get embeddings from backbone
            outputs = model(**inputs)
            
            # Use embedding head if available, otherwise CLS token pooling
            if embedding_head is not None:
                try:
                    # Use the loaded embedding head
                    batch_embeddings = embedding_head(outputs.last_hidden_state, inputs['attention_mask'])
                except Exception as e:
                    print(f"⚠️ Error using embedding head: {e}")
                    print("   Falling back to CLS token pooling")
                    # Fallback to CLS token
                    cls_embedding = outputs.last_hidden_state[:, 0]
                    batch_embeddings = F.normalize(cls_embedding, p=2, dim=1)
            else:
                # Fallback: CLS token pooling (standard BGE-M3 approach)
                print("ℹ️ Using CLS token pooling (BGE-M3 standard)")
                cls_embedding = outputs.last_hidden_state[:, 0]  # CLS token [batch_size, hidden_size]
                batch_embeddings = F.normalize(cls_embedding, p=2, dim=1)
            
            embeddings.append(batch_embeddings.cpu())
    
    return torch.cat(embeddings, dim=0)


def evaluate_sts(model, tokenizer, embedding_head=None, dataset_name="mteb/stsbenchmark-sts"):
    """Evaluate on STS dataset"""
    print(f"Loading STS dataset: {dataset_name}")
    
    # Load dataset - try MTEB format first, fallback to sentence-transformers
    try:
        dataset = load_dataset(dataset_name)
        print("✅ Using MTEB format (consistent with pruning/finetuning)")
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
    embeddings1 = encode_texts(sentences1, model, tokenizer, embedding_head)
    
    print("Encoding sentence 2...")
    embeddings2 = encode_texts(sentences2, model, tokenizer, embedding_head)
    
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


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate BGE-M3 pruned model')
    parser.add_argument('--model-path', type=str, default='experiments/production_hf_masked',
                       help='Path to the pruned model directory')
    args = parser.parse_args()
    
    # Load model with embedding head
    model, tokenizer, embedding_head = load_pruned_model(args.model_path)
    
    # Evaluate on STS
    results = evaluate_sts(model, tokenizer, embedding_head)
    
    print(f"\n✅ Evaluation complete!")
    print(f"Final Spearman correlation: {results['spearman']:.4f}")
    
    if embedding_head is None:
        print("⚠️ WARNING: No embedding head was used - results may be significantly lower than expected!")
        print("   For optimal performance, ensure embedding_heads.pt exists in the model directory.")


if __name__ == "__main__":
    main()
