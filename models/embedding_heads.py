import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class DenseEmbeddingHead(nn.Module):
    """Dense embedding head matching BGE-M3's actual implementation"""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        # BGE-M3 uses CLS token directly without additional projection
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # BGE-M3 style: Use CLS token with L2 normalization
        cls_embedding = hidden_states[:, 0]  # [batch_size, hidden_size]
        return F.normalize(cls_embedding, p=2, dim=-1)

class SparseEmbeddingHead(nn.Module):
    """SPLADE-style sparse head: MLM logits -> log(1+ReLU(logits)) summed over tokens."""

    def __init__(self, hidden_size: int, vocab_size: int = 250002, embedding_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_weight = embedding_weight  # shared word embedding matrix (vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)  # light proj before vocab to allow adaptation

    def set_embedding_weight(self, weight: torch.Tensor):
        self.embedding_weight = weight

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if self.embedding_weight is None:
            raise RuntimeError("SparseEmbeddingHead requires embedding_weight to be set.")

        # Optional lightweight projection then vocab logits (tied weights)
        hs = self.proj(hidden_states)
        logits = torch.matmul(hs, self.embedding_weight.t())  # [B, T, V]
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            logits = logits * mask + (1 - mask) * (-1e4)  # suppress pads

        # SPLADE activation: log(1 + ReLU(logits))
        act = torch.log1p(F.relu(logits))  # [B,T,V]
        # Aggregate (sum over tokens)
        doc_sparse = act.sum(dim=1)  # [B,V]
        # Token-level max for lexical salience (optional)
        token_weights = act.max(dim=-1).values  # [B,T]

        return {
            'lexical_weights': token_weights,
            'sparse_embedding': doc_sparse
        }

class MultiVectorEmbeddingHead(nn.Module):
    """Multi-vector embedding head for BGE-M3 (ColBERT-style)"""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size, bias=False)
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Project and normalize all token representations
        multi_vector = self.dense(hidden_states)
        multi_vector = F.normalize(multi_vector, p=2, dim=-1)
        
        if attention_mask is not None:
            # Zero out padded positions
            multi_vector = multi_vector * attention_mask.unsqueeze(-1).float()
        
        return multi_vector


class BGEEmbeddingHeads(nn.Module):
    """BGE-M3 compatible embedding heads"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.vocab_size = getattr(config, 'vocab_size', 250002)
        self.dense_head = DenseEmbeddingHead(self.hidden_size)
        self.sparse_head = SparseEmbeddingHead(self.hidden_size, self.vocab_size)
        self.multi_vector_head = MultiVectorEmbeddingHead(self.hidden_size)
        
    def forward(self, 
                hidden_states: torch.Tensor,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                return_dense: bool = True,
                return_sparse: bool = False,
                return_multi_vector: bool = False) -> Dict[str, torch.Tensor]:
        
        outputs = {}
        
        if return_dense:
            outputs['dense_embedding'] = self.dense_head(hidden_states, attention_mask)
        
        if return_sparse:
            sparse_outputs = self.sparse_head(hidden_states, input_ids, attention_mask)
            outputs.update(sparse_outputs)
        
        if return_multi_vector:
            outputs['colbert_vecs'] = self.multi_vector_head(hidden_states, attention_mask)
        
        return outputs
    def load_from_pretrained_bge(self, pretrained_model):
        """Load weights from pretrained BGE-M3 model"""
        # This would be implemented to copy weights from the original model
        # For now, the heads start from scratch
        pass

    def set_sparse_embedding_weight_from_backbone(self, backbone):
        if hasattr(backbone, 'embeddings') and hasattr(backbone.embeddings, 'word_embeddings'):
            self.sparse_head.set_embedding_weight(backbone.embeddings.word_embeddings.weight)