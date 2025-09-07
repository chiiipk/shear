import torch
import torch.nn as nn
import torch.nn.functional as F
from composer.models.base import ComposerModel
from composer.utils import dist, get_device, reproducibility
from omegaconf import DictConfig
from typing import Dict, Optional, Any, Tuple
from transformers import AutoModel, AutoConfig, AutoTokenizer

from .l0_module_embedding import L0ModuleEmbedding


from .embedding_heads import BGEEmbeddingHeads
from .bge_m3_backbone import MaskedBGEM3Backbone

class ComposerBGEM3(ComposerModel):
    """BGE-M3 model with L0 pruning and Composer interface"""
    
    def __init__(self, cfg):
        super().__init__()
        
        # Load pretrained BGE-M3 model and config
        model_name = getattr(cfg, 'base_model', 'BAAI/bge-m3')
        self.base_model_name = model_name  # Store for HF export
        base_model = AutoModel.from_pretrained(model_name)
        self.config = base_model.config
        
        # Create masked backbone with original config
        self.backbone = MaskedBGEM3Backbone(self.config)
        self.backbone.load_state_dict(base_model.state_dict(), strict=False)
        
        # Store original config before any modifications
        self.original_config = base_model.config
        
        # Free memory - we don't need the full original model anymore
        del base_model
        
        # Override config with custom settings if provided
        if hasattr(cfg, 'd_model'):
            self.config.hidden_size = cfg.d_model
        if hasattr(cfg, 'n_layers'):
            self.config.num_hidden_layers = cfg.n_layers
        if hasattr(cfg, 'n_heads'):
            self.config.num_attention_heads = cfg.n_heads
            
        # Initialize embedding heads with BGE-M3 compatible approach
        self.embedding_heads = BGEEmbeddingHeads(self.config)
        # Tie sparse head vocab projection to backbone embeddings early (safe even if unused)
        self.embedding_heads.set_sparse_embedding_weight_from_backbone(self.backbone)
        
        # Fix device string conversion
        device_name = get_device(None).name
        if device_name == "gpu":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"

        # Validate configuration consistency
        self._validate_config()

        # Initialize L0 module for pruning with model info
        self.l0_module = L0ModuleEmbedding(cfg, device_name, self.backbone)

        # Loss / weighting configuration
        self.temperature = getattr(cfg, 'temperature', 0.02)
        self.w_dense = getattr(cfg, 'w_dense', 1.0)
        self.w_sparse = getattr(cfg, 'w_sparse', 0.0)
        self.w_multi = getattr(cfg, 'w_multi', 0.0)
        self.w_agree = getattr(cfg, 'w_agree', 0.0)
        self.sparse_l1 = getattr(cfg, 'sparse_l1', 0.0)
        self.sparse_flops = getattr(cfg, 'sparse_flops', 0.0)
        self.enable_sparse = self.w_sparse > 0
        self.enable_multi = self.w_multi > 0
        # Configurable L0 loss schedule
        self.l0_weight_warmup = getattr(cfg, 'l0_weight_warmup', 100)
        self.l0_weight_max = getattr(cfg, 'l0_weight_max', 10.0)
        
        # Step counter for L0 module warmup (fix for lagrangian warmup)
        self.global_step_counter = 0
        
        # Metrics storage
        self.train_metrics = {}
        self.eval_metrics = {}
        self.ref_model = None
        
    def forward(self, batch):
        """Forward pass through the model"""
        input_ids = batch['input_ids']
        attention_mask = batch.get('attention_mask', None)
        
        # Increment global step counter for L0 module warmup
        if self.training:
            self.global_step_counter += 1
        
        # Get current training step for L0 module
        current_step = self.global_step_counter
        if hasattr(self.l0_module, 'update_training_step'):
            self.l0_module.update_training_step(current_step)
        
        # Get L0 masks - CRITICAL: pass step for dynamic sparsity
        l0_masks = self.l0_module(pruned_steps=current_step)
        
        # CRITICAL: Also get Lagrangian loss if in training mode
        l0_loss = None
        if self.training:
            l0_loss, l0_metrics = self.l0_module(calculate_lagrangian=True, pruned_steps=current_step)
        
        # Forward through backbone with soft masking
        backbone_outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            layer_z=l0_masks.get('layer_z'),
            head_z=l0_masks.get('head_z'),
            intermediate_z=l0_masks.get('intermediate_z'),
        )
        
        # Generate embeddings
        embedding_outputs = self.embedding_heads(
            hidden_states=backbone_outputs["last_hidden_state"],
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dense=True,
            return_sparse=self.enable_sparse,
            return_multi_vector=self.enable_multi,
        )
        
        return {
            'embeddings': embedding_outputs,
            'l0_loss': l0_loss,  # CRITICAL: Return L0 loss
            'l0_masks': l0_masks,
            'backbone_outputs': backbone_outputs,
        }

    def loss(self, outputs, batch):
        """Simplified loss computation"""
        embeddings = outputs['embeddings']
        l0_loss = outputs.get('l0_loss')

        batch_pairs = batch['input_ids'].size(0) // 2

        # Dense branch (contrastive + optional regression if similarity scores present)
        dense_contrastive = self.compute_contrastive_loss(embeddings, batch_pairs)
        dense_reg = 0.0
        if 'similarity_scores' in batch:
            dense_reg = self.compute_sts_regression(embeddings, batch, batch_pairs)
            # Add lightweight contrastive supervision from high/low score threshold
            sts_contrastive = self.compute_sts_contrastive(embeddings, batch, batch_pairs)
            dense_contrastive = (dense_contrastive + sts_contrastive) * 0.5
        dense_loss = dense_contrastive + dense_reg

        # Sparse branch
        sparse_loss = 0.0
        if self.enable_sparse and 'sparse_embedding' in embeddings:
            sparse_loss = self.compute_sparse_loss(embeddings)

        # Multi-vector branch
        multi_loss = 0.0
        if self.enable_multi and 'colbert_vecs' in embeddings:
            multi_loss = self.compute_multivector_loss(embeddings, batch_pairs)

        # Agreement regularizer (simple cosine distribution alignment between dense and others)
        agree_loss = 0.0
        if self.w_agree > 0 and (self.enable_sparse or self.enable_multi):
            with torch.no_grad():
                dense_vec = embeddings['dense_embedding']  # [2B, D]
            parts = []
            if self.enable_sparse and 'sparse_embedding' in embeddings:
                # Project sparse to dense space proxy by cosine with dense vec (self-sim) – lightweight proxy
                se = embeddings['sparse_embedding']
                se_norm = se / (se.norm(dim=-1, keepdim=True) + 1e-8)
                dv_norm = dense_vec / (dense_vec.norm(dim=-1, keepdim=True) + 1e-8)
                parts.append(1 - (dv_norm * se_norm[:, :dv_norm.size(1)].detach()).sum(-1).mean())
            if self.enable_multi and 'colbert_vecs' in embeddings:
                mv = embeddings['colbert_vecs']
                cls = mv[:, 0]
                cls_norm = cls / (cls.norm(dim=-1, keepdim=True) + 1e-8)
                dv_norm = dense_vec / (dense_vec.norm(dim=-1, keepdim=True) + 1e-8)
                parts.append(1 - (cls_norm * dv_norm).sum(-1).mean())
            if parts:
                agree_loss = sum(parts) / len(parts)

        total_loss = (
            self.w_dense * dense_loss +
            self.w_sparse * sparse_loss +
            self.w_multi * multi_loss +
            self.w_agree * agree_loss
        )
        
        # Add L0 constraint loss with dynamic weighting
        if l0_loss is not None:
            # More aggressive schedule: short warmup, strong max weight
            warmup_steps = self.l0_weight_warmup
            max_weight = self.l0_weight_max
            current_step = self.global_step_counter
            if current_step < warmup_steps:
                l0_weight = 0.0
            else:
                l0_weight = min((current_step - warmup_steps) / 500.0, 1.0) * max_weight
            total_loss = total_loss + l0_weight * l0_loss
        
        return total_loss

    
    def compute_sts_regression(self, embeddings: Dict[str, torch.Tensor], batch: Dict[str, Any], batch_size: int) -> torch.Tensor:
        dense_emb = embeddings['dense_embedding']
        scores = batch['similarity_scores']
        s1 = dense_emb[0::2]
        s2 = dense_emb[1::2]
        sim = F.cosine_similarity(s1, s2, dim=-1)
        sim = (sim + 1) * 2.5
        return F.mse_loss(sim, scores)
    
    def compute_contrastive_loss(self, embeddings: Dict[str, torch.Tensor], batch_size: int) -> torch.Tensor:
        """Production contrastive loss for query-passage pairs"""
        dense_emb = embeddings['dense_embedding']  # [batch_size * 2, embedding_dim]
        
        # Extract queries and passages from interleaved format
        query_emb = dense_emb[0::2]  # Even indices: queries [batch_size, embedding_dim]
        passage_emb = dense_emb[1::2]  # Odd indices: passages [batch_size, embedding_dim]
        
        # Compute similarity matrix: queries vs all passages
        similarity_matrix = torch.matmul(query_emb, passage_emb.t()) / self.temperature
        
        # Labels: each query should match its corresponding passage (diagonal)
        labels = torch.arange(batch_size, device=query_emb.device)
        
        # InfoNCE loss
        contrastive_loss = F.cross_entropy(similarity_matrix, labels)
        return contrastive_loss

    def compute_sparse_loss(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Simple sparse L1 + proxy contrast using lexical weights self-sim (placeholder minimal yet final approach)
        w = embeddings['lexical_weights']  # [B*seq]
        l1 = w.mean()
        # FLOPs surrogate = log(1+ w)
        fl = torch.log1p(w).mean()
        return self.sparse_l1 * l1 + self.sparse_flops * fl

    def compute_multivector_loss(self, embeddings: Dict[str, torch.Tensor], batch_size: int) -> torch.Tensor:
        vecs = embeddings['colbert_vecs']  # [2B, T, D]
        q = vecs[0::2]
        p = vecs[1::2]
        # Late interaction: score(i,j)= sum_t max_{t'} dot(q_t, p_{t'})
        # Compute all pair scores (B,B)
        # q: [B,T,D], p: [B,T,D]
        B, T, D = q.size()
        # Normalize (already normalized but ensure numerical stability)
        qn = q
        pn = p
        # Expand for pairwise: qn -> [B,1,T,D]; pn -> [1,B,T,D]
        q_exp = qn.unsqueeze(1)
        p_exp = pn.unsqueeze(0)  # [1,B,T,D]
        # Dot: [B,B,T,T]
        dots = torch.einsum('bqtd, pbtd -> bqtp', q_exp, p_exp)
        # Max over passage tokens: [B,B,T]
        max_tp = dots.max(dim=-1).values
        scores = max_tp.sum(dim=-1) / (T + 1e-6)  # normalize by length
        scores = scores / self.temperature
        labels = torch.arange(B, device=vecs.device)
        return F.cross_entropy(scores, labels)

    def compute_sts_contrastive(self, embeddings: Dict[str, torch.Tensor], batch: Dict[str, Any], batch_size: int) -> torch.Tensor:
        scores = batch['similarity_scores']  # [B]
        dense_emb = embeddings['dense_embedding']
        q = dense_emb[0::2]
        p = dense_emb[1::2]
        # positives indices where score >=4, negatives <=2
        pos_mask = (scores >= 4.0)
        neg_mask = (scores <= 2.0)
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            return torch.zeros([], device=dense_emb.device)
        pos_q = q[pos_mask]
        pos_p = p[pos_mask]
        neg_p = p[neg_mask]
        # Build similarity: each pos query vs its true pos + all neg passages
        sim_pos = (pos_q @ pos_p.T) / self.temperature  # should align diagonal
        sim_neg = (pos_q @ neg_p.T) / self.temperature
        # Compose logits: true pairs on diagonal of first block
        logits = torch.cat([sim_pos, sim_neg], dim=1)
        labels = torch.arange(pos_q.size(0), device=dense_emb.device)
        return F.cross_entropy(logits, labels)
    
    def compute_constraint_loss(self, expected_sparsity: Dict[str, torch.Tensor]) -> torch.Tensor:
        """LLM-Shearing style constraint loss with warmup and quadratic penalty"""
        constraint_loss = 0.0
        
        for mask_name, sparsity in expected_sparsity.items():
            if mask_name in self.l0_module.masks:
                mask = self.l0_module.masks[mask_name]
                
                if hasattr(mask, 'target_sparsity') and mask.target_sparsity is not None:
                    # Get warmup target sparsity (gradually increases from 0 to final target)
                    current_target = self.l0_module.get_warmup_target_sparsity(mask.target_sparsity)
                    
                    # LLM-Shearing style: Linear + Quadratic penalty
                    sparsity_diff = sparsity.mean() - current_target
                    linear_penalty = torch.abs(sparsity_diff)
                    quadratic_penalty = sparsity_diff ** 2
                    
                    # Combine linear + quadratic (quadratic is stronger for large violations)
                    constraint_loss += linear_penalty + 5.0 * quadratic_penalty
        
        return constraint_loss
    
    def get_metrics(self, is_train: bool = False) -> Dict[str, Any]:
        """Get metrics for logging"""
        if is_train:
            return self.train_metrics
        else:
            return self.eval_metrics
    
    def prune_params(self, zs: Optional[Dict[str, torch.Tensor]] = None):
        """Prune model parameters based on masks"""
        if zs is None:
            zs = self.l0_module()
        
        # Prune backbone
        self.backbone.prune_params(zs)
        
        # Note: Hidden dimension pruning removed for production version
    
    def get_model_info(self):
        """Get model architecture information"""
        return {
            'base_model_info': self.l0_module.base_model_info,
            'target_model_info': self.l0_module.target_model_info,
            'pruning_modules': self.l0_module.pruning_modules,
        }
    
    def reset_step_counter(self):
        """Reset the global step counter to 0"""
        self.global_step_counter = 0
    
    def set_step_counter(self, step: int):
        """Set the global step counter to a specific value (useful for checkpoint resuming)"""
        self.global_step_counter = step
    
    def sync_step_with_composer_state(self, state):
        """Sync step counter with Composer state (call from callback if needed)"""
        if hasattr(state, 'timestamp') and hasattr(state.timestamp, 'batch'):
            self.global_step_counter = state.timestamp.batch.value
    
    def get_step_counter(self):
        """Get the current global step counter value"""
        return self.global_step_counter
    
    def compute_spearman_correlation(self, predicted_scores: torch.Tensor, 
                                   ground_truth_scores: torch.Tensor) -> float:
        """Compute Spearman correlation for STS evaluation"""
        try:
            from scipy.stats import spearmanr
            pred_np = predicted_scores.detach().cpu().numpy()
            gt_np = ground_truth_scores.detach().cpu().numpy()
            correlation, _ = spearmanr(pred_np, gt_np)
            return float(correlation)
        except ImportError:
            # Fallback to Pearson correlation if scipy not available
            pred_centered = predicted_scores - predicted_scores.mean()
            gt_centered = ground_truth_scores - ground_truth_scores.mean()
            correlation = (pred_centered * gt_centered).sum() / (
                torch.sqrt((pred_centered ** 2).sum() * (gt_centered ** 2).sum()) + 1e-8
            )
            return float(correlation)
    
    def extract_pruned_model(self) -> 'ComposerBGEM3':
        """Extract a pruned model with parameters permanently removed"""
        # Get current masks
        zs = self.l0_module()
        
        # Create new config based on pruned dimensions
        pruned_config = self._create_pruned_config(zs)
        
        # Create new model with pruned config
        pruned_model = ComposerBGEM3(pruned_config)
        
        # Copy and prune weights
        self._copy_pruned_weights(pruned_model, zs)
        
        return pruned_model
    
    def _create_pruned_config(self, zs: Dict[str, torch.Tensor]) -> DictConfig:
        """Create configuration for pruned model"""
        # This would create a new config with reduced dimensions
        # Implementation depends on specific pruning strategy
        pass
    
    def _copy_pruned_weights(self, target_model: 'ComposerBGEM3', 
                           zs: Dict[str, torch.Tensor]):
        """Copy weights from current model to pruned model"""
        # This would copy only the non-pruned weights
        # Implementation depends on specific pruning strategy
        pass
    
    def save_pruned_hf_model(self, save_path: str, tokenizer_name: str = None, export_mode: str = "structural"):
        """Save pruned model in HuggingFace format for production use
        
        Args:
            save_path: Path to save the model
            tokenizer_name: Base model name for tokenizer (optional)
            export_mode: Either "structural" (default) or "masked"
                - structural: Export with actually pruned dimensions (current behavior)
                - masked: Export with original dimensions but zero out pruned weights
        """
        import sys
        import os
        from pathlib import Path
        
        # Add project root to path for imports
        project_root = Path(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        from utils.hf_export import export_pruned_backbone_clean, export_masked_backbone
        import json
        
        # Use eval mode for deterministic masks
        was_training = self.training
        self.eval()
        
        # Get deterministic masks using final training step
        final_step = self.global_step_counter
        print(f"\n🎯 Export mode: {export_mode}")
        print(f"🎯 Using final training step: {final_step}")
        
        # Use dedicated export method for consistent masks
        zs = self.l0_module.get_export_masks()
        print("🎯 Export masks:")
        for mask_name, mask_tensor in zs.items():
            sparsity = (mask_tensor == 0).float().mean().item()
            print(f"  {mask_name}: {sparsity:.1%} sparsity")
        
        base_model_name = tokenizer_name or self.base_model_name
        
        if export_mode == "masked":
            # Export with original dimensions but masked weights
            print(f"\n💾 Saving masked model to {save_path}")
            export_masked_backbone(self.backbone, save_path, base_model_name, self.embedding_heads, zs)
        else:
            # Default: structural pruning (original behavior)
            print("\n🎯 Applying structural pruning...")
            
            # Actually remove pruned parameters
            self.prune_params(zs)
            self._validate_pruned_model()
            
            # Save backbone using clean export (no padding)
            print(f"\n💾 Saving structurally pruned model to {save_path}")
            export_pruned_backbone_clean(self.backbone, save_path, base_model_name, self.embedding_heads)
        
        # Save pruning info
        pruning_info = {
            'export_mode': export_mode,
            'pruning_results': {name: float((mask == 0).float().mean()) for name, mask in zs.items()},
            'base_model': base_model_name,
        }
        
        if export_mode == "masked":
            # For masked export, show original dimensions
            pruning_info['final_config'] = {
                'num_hidden_layers': self.config.num_hidden_layers,
                'num_attention_heads': self.config.num_attention_heads,
                'intermediate_size': self.config.intermediate_size,
                'hidden_size': self.config.hidden_size,
                'note': 'Original dimensions preserved, pruned weights zeroed'
            }
        else:
            # For structural export, show actual pruned dimensions
            pruning_info['final_config'] = {
                'num_hidden_layers': len(self.backbone.encoder.layer),
                'num_attention_heads': self.backbone.encoder.layer[0].attention.num_attention_heads if len(self.backbone.encoder.layer) > 0 else 0,
                'intermediate_size': self.backbone.encoder.layer[0].intermediate.dense.out_features if len(self.backbone.encoder.layer) > 0 else 0,
                'hidden_size': self.config.hidden_size,
                'note': 'Structurally pruned dimensions'
            }
        
        with open(os.path.join(save_path, 'pruning_info.json'), 'w') as f:
            json.dump(pruning_info, f, indent=2)
        
        print(f"✅ Clean pruned model saved in HuggingFace format!")
        print(f"📁 Location: {save_path}")
        print(f"🔧 Usage: model = AutoModel.from_pretrained('{save_path}')")
        
        # Restore training mode
        if was_training:
            self.train()
        
        return save_path
    
    def _validate_config(self):
        """Validate model configuration for mathematical consistency"""
        if self.config.hidden_size % self.config.num_attention_heads != 0:
            raise ValueError(
                f"Hidden size ({self.config.hidden_size}) must be divisible by "
                f"number of attention heads ({self.config.num_attention_heads}). "
                f"Adjust configuration to use valid combinations."
            )
    
    def _validate_pruned_model(self):
        """Validate pruned model is in correct state for production use"""
        # Validate backbone configuration
        backbone_config = self.backbone.config
        
        # Check layer count consistency
        actual_layers = len(self.backbone.encoder.layer)
        config_layers = backbone_config.num_hidden_layers
        if actual_layers != config_layers:
            raise ValueError(f"Layer count mismatch: actual={actual_layers}, config={config_layers}")
        
        # Check attention head consistency
        if backbone_config.hidden_size % backbone_config.num_attention_heads != 0:
            raise ValueError(f"Invalid attention configuration after pruning: "
                           f"hidden_size={backbone_config.hidden_size}, "
                           f"num_attention_heads={backbone_config.num_attention_heads}")
        
        print(f"✅ Model validation passed: {actual_layers} layers, "
              f"{backbone_config.num_attention_heads} heads, "
              f"{backbone_config.intermediate_size} intermediate size")
