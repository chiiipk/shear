import math
import torch
from torch import nn
import torch.nn.functional as F
from argparse import Namespace as NS
from typing import Any, List, Dict
from composer.core.time import Time
from torch.autograd import Variable

limit_a, limit_b, epsilon = -.1, 1.1, 1e-6

class Mask(nn.Module):
    def __init__(self, 
                 name: str,
                 mask_shape: List, 
                 num_params_per_mask: int, 
                 mask_output_shape: List, 
                 target_sparsity: float,
                 target_mask_size: int,
                 device: str,
                 eval_target_model: bool=True,
                 sparsity_warmup_steps: int = 1000,
                 sparsity_anneal_steps: int = 5000) -> None:
        super().__init__()
        self.name = name
        self.num_params_per_mask = num_params_per_mask
        self.mask_output_shape = mask_output_shape
        self.target_sparsity = target_sparsity
        
        # Hard concrete distribution parameters (from original paper)
        self.droprate_init = 0.5
        self.temperature = 2./3.
        self.magical_number = 0.8
        self.device = device
        
        self.z_loga = self.initialize_mask(mask_shape) 
        self.mask_size = self.z_loga.shape[-1]
        self.target_mask_size = target_mask_size
        self.eval_target_model = eval_target_model
        
        # Dynamic sparsity scheduling parameters from config
        self.warmup_steps = sparsity_warmup_steps
        self.anneal_steps = sparsity_anneal_steps
        
    def target_frac(self, step: int, final_frac: float, warmup: int = None, anneal: int = None):
        """Compute target sparsity fraction based on current step"""
        warmup = warmup or self.warmup_steps
        anneal = anneal or self.anneal_steps
        
        if step < warmup:
            return 0.0  # No pruning early
        
        prog = min(1.0, (step - warmup) / max(1, anneal))
        return final_frac * prog  # Linear ramp
        
    def param_init_fn(self, module):
        """Initialize parameters as in original LLM-Shearing"""
        mean = math.log(1 - self.droprate_init) - math.log(self.droprate_init)
        if isinstance(module, nn.Parameter):
            module.data.normal_(mean, 1e-2)
        else:
            for tensor in module.parameters():
                tensor.data.normal_(mean, 1e-2)
        
    def initialize_mask(self, mask_shape: List):
        z_loga = nn.Parameter(torch.ones(*mask_shape, device=self.device))
        self.param_init_fn(z_loga)
        return z_loga

    def cdf_qz(self, z_loga: torch.Tensor = None):
        """CDF of stretched concrete distribution"""
        if z_loga is None:
            z_loga = self.z_loga
        xn = (0 - limit_a) / (limit_b - limit_a)
        logits = math.log(xn) - math.log(1 - xn)
        return torch.sigmoid(logits * self.temperature - z_loga).clamp(min=epsilon, max=1 - epsilon)
    
    def get_eps(self, size):
        """Uniform random numbers for the concrete distribution"""
        eps = torch.empty(size, device=self.z_loga.device).uniform_(epsilon, 1-epsilon)
        return eps

    def quantile_concrete(self, eps: torch.Tensor):
        """Quantile function for concrete distribution"""
        # Ensure eps and z_loga have compatible shapes
        assert eps.shape == self.z_loga.shape, f"Shape mismatch: eps {eps.shape} vs z_loga {self.z_loga.shape}"
        y = torch.sigmoid((torch.log(eps) - torch.log(1 - eps) + self.z_loga) / self.temperature)
        return y * (limit_b - limit_a) + limit_a

    def sample_z(self, step: int = 0):
        """Unified stochastic mask sampling.

        Training: Gumbel-Concrete for better exploration, straight-through to keep gradients.
        Eval: deterministic per-layer (or global for 1D) top-k if target_mask_size is set; else 0.5 threshold.
        Export shares identical logic with eval (see sample_z_for_export) to eliminate mismatch.
        """
        soft_mask = torch.sigmoid(self.z_loga / self.temperature * self.magical_number)
        if self.training:
            # Gumbel noise
            eps = torch.rand_like(soft_mask).clamp_(1e-6, 1 - 1e-6)
            gumbel = -torch.log(-torch.log(eps))
            gumbel_mask = torch.sigmoid((self.z_loga + gumbel) / (self.temperature))
            if self.target_mask_size is not None and self.target_mask_size > 0 and gumbel_mask.numel() > 1:
                # Per-layer (row-wise) top-k for 2D, global top-k for 1D
                if gumbel_mask.dim() == 2:
                    k = min(self.target_mask_size, gumbel_mask.size(1))
                    values, indices = gumbel_mask.topk(k=k, dim=1, largest=True)
                    hard = torch.zeros_like(gumbel_mask)
                    hard.scatter_(1, indices, 1.0)
                elif gumbel_mask.dim() == 1:
                    k = min(self.target_mask_size, gumbel_mask.size(0))
                    values, indices = gumbel_mask.topk(k=k, dim=0, largest=True)
                    hard = torch.zeros_like(gumbel_mask)
                    hard.scatter_(0, indices, 1.0)
                else:
                    hard = (gumbel_mask > 0.5).float()
            else:
                hard = (gumbel_mask > 0.5).float()
            st = hard - gumbel_mask.detach() + gumbel_mask  # straight-through
            return st.reshape(*self.mask_output_shape)
        else:
            if self.target_mask_size is not None and self.target_mask_size > 0 and soft_mask.numel() > 1:
                if soft_mask.dim() == 2:
                    k = min(self.target_mask_size, soft_mask.size(1))
                    values, indices = soft_mask.topk(k=k, dim=1, largest=True)
                    hard = torch.zeros_like(soft_mask)
                    hard.scatter_(1, indices, 1.0)
                elif soft_mask.dim() == 1:
                    k = min(self.target_mask_size, soft_mask.size(0))
                    values, indices = soft_mask.topk(k=k, dim=0, largest=True)
                    hard = torch.zeros_like(soft_mask)
                    hard.scatter_(0, indices, 1.0)
                else:
                    hard = (soft_mask > 0.5).float()
            else:
                hard = (soft_mask > 0.5).float()
            return hard.reshape(*self.mask_output_shape)
    
    def sample_z_for_export(self):
        """Deterministic export identical to eval branch: per-layer top-k or threshold.

        Guarantees export cardinality matches training-time hard selections.
        """
        soft_mask = torch.sigmoid(self.z_loga / self.temperature * self.magical_number)
        if self.target_mask_size is not None and self.target_mask_size > 0 and soft_mask.numel() > 1:
            if soft_mask.dim() == 2:
                k = min(self.target_mask_size, soft_mask.size(1))
                _, indices = soft_mask.topk(k=k, dim=1, largest=True)
                hard = torch.zeros_like(soft_mask)
                hard.scatter_(1, indices, 1.0)
            elif soft_mask.dim() == 1:
                k = min(self.target_mask_size, soft_mask.size(0))
                _, indices = soft_mask.topk(k=k, dim=0, largest=True)
                hard = torch.zeros_like(soft_mask)
                hard.scatter_(0, indices, 1.0)
            else:
                hard = (soft_mask > 0.5).float()
        else:
            hard = (soft_mask > 0.5).float()
        return hard.reshape(*self.mask_output_shape)

    
    def forward(self, step: int = 0):
        """Forward pass with step-aware sampling"""
        return self.sample_z(step)
    
    def get_soft_mask(self):
        """Get soft mask for sparsity computation"""
        return torch.sigmoid(self.z_loga / self.temperature * self.magical_number)

    def expected_active(self):
        """Return expected active counts (per-row for 2D, scalar for 1D)."""
        sm = self.get_soft_mask()
        if sm.dim() == 2:
            return sm.sum(dim=1)  # per-row counts
        return sm.sum()  # scalar
    
    def constrain_parameters(self):
        """Constrain parameters to valid ranges"""
        pass  # No constraints needed for this implementation
    
    def constrain_parameters(self):
        self.z_loga.data.clamp_(min=math.log(1e-2), max=math.log(1e2))

    def calculate_expected_score_sparsity(self):
        # Return per-element keep scores and per-row sparsity (for 2D) or scalar sparsity (1D)
        score = 1 - self.cdf_qz()
        sparsity = 1 - score.sum(-1) / self.mask_size if score.dim() > 1 else 1 - score.mean()
        return score, sparsity

class L0ModuleEmbedding(nn.Module):
    def __init__(self, cfg, device, bge_model=None):
        super(L0ModuleEmbedding, self).__init__()

        # Extract model info from actual BGE model
        if bge_model is not None:
            self.base_model_info = self.extract_bge_model_info(bge_model)
        else:
            # Fallback to config-based info
            self.base_model_info = self.set_model_info_bge(cfg)
            
        l0_module_cfg = cfg.l0_module
        self.target_model_info = None
        target_model_cfg = getattr(l0_module_cfg, "target_model", None)
        if target_model_cfg is not None:
            self.target_model_info = self.set_model_info_bge(target_model_cfg)
        
        # BGE-M3 specific pruning modules
        self.pruning_modules = l0_module_cfg.pruning_modules        
        self.start_sparsity = l0_module_cfg.start_sparsity 
        self.lagrangian_warmup_steps = Time.from_timestring(l0_module_cfg.lagrangian_warmup_steps).value
        
        # Dynamic sparsity scheduling from config
        self.sparsity_warmup_steps = getattr(l0_module_cfg, 'sparsity_warmup_steps', 1000)
        self.sparsity_anneal_steps = getattr(l0_module_cfg, 'sparsity_anneal_steps', 5000)
        if isinstance(self.sparsity_warmup_steps, str):
            self.sparsity_warmup_steps = Time.from_timestring(self.sparsity_warmup_steps).value
        if isinstance(self.sparsity_anneal_steps, str):
            self.sparsity_anneal_steps = Time.from_timestring(self.sparsity_anneal_steps).value
            
        self.device = device
        self.eval_target_model = l0_module_cfg.get("eval_target_model", True)
        
        # Embedding model specific parameters
        self.embedding_loss_weight = l0_module_cfg.get("embedding_loss_weight", 1.0)
        self.contrastive_loss_weight = l0_module_cfg.get("contrastive_loss_weight", 1.0)
        
        # l0 params
        self.lambdas = {}
        self.lambdas["lambda_1"] = torch.nn.Parameter(torch.tensor(0.0, device=device))
        self.lambdas["lambda_2"] = torch.nn.Parameter(torch.tensor(0.0, device=device))
        self.masks = {}
        
        for pruning_module in self.pruning_modules:
            self.initialize_one_module(pruning_module)
        
        self.masks = torch.nn.ModuleDict(self.masks)
        self.lambdas = torch.nn.ParameterDict(self.lambdas)
        
        # Calculate model sizes
        self.prunable_model_size = self.calculate_prunable_model_size(self.base_model_info)
        if target_model_cfg is not None:
            self.prunable_target_model_size = self.calculate_prunable_model_size(self.target_model_info)
            self.target_sparsity = 1 - self.prunable_target_model_size / self.prunable_model_size
        else:
            self.target_sparsity = l0_module_cfg.target_sparsity

        # Step tracking for dynamic sparsity
        self.current_step = 0
        self.export_mode = False  # Flag for deterministic export

        print("********** Initializing L0 Module for BGE-M3 **********") 
        print(f"Dynamic sparsity scheduling:")
        print(f"  Sparsity warmup steps: {self.sparsity_warmup_steps}")
        print(f"  Sparsity anneal steps: {self.sparsity_anneal_steps}")
        print(f"  Lagrangian warmup steps: {self.lagrangian_warmup_steps}")
        for pruning_module in self.pruning_modules:
            print(f"***** {pruning_module} *****")
            print(f"z.shape", self.masks[pruning_module].z_loga.shape)
            print(f"size", self.masks[pruning_module].mask_size)
        print(f"prunable model size: {self.prunable_model_size}")
        print(f"target sparsity: {self.target_sparsity}")

    def extract_bge_model_info(self, bge_model):
        """Extract model information from actual BGE model"""
        config = bge_model.config
        ns = NS()
        
        # XLM-RoBERTa specific parameters
        ns.hidden_size = config.hidden_size  # 1024 for BGE-M3
        ns.intermediate_size = config.intermediate_size  # 4096 for BGE-M3  
        ns.num_attention_heads = config.num_attention_heads  # 16 for BGE-M3
        ns.num_hidden_layers = config.num_hidden_layers  # 24 for BGE-M3
        ns.dim_per_head = ns.hidden_size // ns.num_attention_heads
        ns.vocab_size = config.vocab_size
        ns.max_position_embeddings = config.max_position_embeddings
        
        # BGE-M3 has multiple output heads
        ns.has_dense_head = True
        ns.has_sparse_head = True  
        ns.has_colbert_head = True
        
        # Parameter calculations for XLM-RoBERTa
        # Each layer has: Q,K,V,O projections + 2 MLP layers + 2 LayerNorms
        ns.params_per_head = ns.hidden_size * ns.dim_per_head  # Per head parameters
        ns.params_per_layer = (
            4 * ns.hidden_size * ns.hidden_size +  # Q,K,V,O projections
            2 * ns.hidden_size * ns.intermediate_size +  # MLP up/down
            4 * ns.hidden_size  # 2 LayerNorms + bias terms
        )
        ns.params_per_intermediate_dim = 2 * ns.hidden_size  # Up and down projection per dim
        
        return ns
    
    def set_model_info_bge(self, cfg):
        """Set model info from config for BGE-M3"""
        ns = NS()
        # Accept both canonical and legacy keys (n_layers/n_heads) for robustness
        ns.hidden_size = getattr(cfg, 'hidden_size', 1024)
        ns.intermediate_size = getattr(cfg, 'intermediate_size', 4096)
        ns.num_attention_heads = getattr(cfg, 'num_attention_heads', getattr(cfg, 'n_heads', 16))
        ns.num_hidden_layers = getattr(cfg, 'num_hidden_layers', getattr(cfg, 'n_layers', 24))
        ns.dim_per_head = ns.hidden_size // ns.num_attention_heads
        ns.vocab_size = getattr(cfg, 'vocab_size', 250002)

        # BGE-M3 specific
        ns.has_dense_head = True
        ns.has_sparse_head = True
        ns.has_colbert_head = True

        ns.params_per_head = ns.hidden_size * ns.dim_per_head
        ns.params_per_layer = (
            4 * ns.hidden_size * ns.hidden_size +
            2 * ns.hidden_size * ns.intermediate_size +
            4 * ns.hidden_size
        )
        ns.params_per_intermediate_dim = 2 * ns.hidden_size

        return ns
        
    def calculate_prunable_model_size(self, ns: NS):
        """Calculate total prunable parameters"""
        prunable_model_size = 0
        
        if "head" in self.pruning_modules:
            prunable_model_size += ns.params_per_head * ns.num_attention_heads * ns.num_hidden_layers
            
        if "layer" in self.pruning_modules:
            prunable_model_size += ns.params_per_layer * ns.num_hidden_layers
            
        if "intermediate" in self.pruning_modules:
            prunable_model_size += ns.params_per_intermediate_dim * ns.intermediate_size * ns.num_hidden_layers
            
        return prunable_model_size
        
    def initialize_one_module(self, module_name: str):
        func_name = f"initialize_{module_name}"
        try:
            method = getattr(self, func_name)
        except AttributeError:
            raise NotImplementedError("Instance `{}` does not implement `{}`".format(self, func_name))
        method()

    def initialize_head(self):
        """Initialize attention head pruning masks"""
        mask_shape = [self.base_model_info.num_hidden_layers, self.base_model_info.num_attention_heads]
        num_params_per_mask = self.base_model_info.params_per_head
        mask_output_shape = [self.base_model_info.num_hidden_layers, 1, self.base_model_info.num_attention_heads, 1] 
        
        target_head_sparsity = None
        target_mask_size = None
        if self.target_model_info is not None:
            target_head_sparsity = 1 - self.target_model_info.num_attention_heads / self.base_model_info.num_attention_heads
            target_mask_size = self.target_model_info.num_attention_heads
            pd = {"lambda_1_head": torch.nn.Parameter(torch.tensor(0.0, device=self.device)),
                  "lambda_2_head": torch.nn.Parameter(torch.tensor(0.0, device=self.device))}
            self.lambdas.update(pd)
        
        head_mask = Mask(name="head",
                         mask_shape=mask_shape,
                         num_params_per_mask=num_params_per_mask,
                         mask_output_shape=mask_output_shape,
                         target_sparsity=target_head_sparsity,
                         target_mask_size=target_mask_size,
                         device=self.device,
                         eval_target_model=self.eval_target_model,
                         sparsity_warmup_steps=self.sparsity_warmup_steps,
                         sparsity_anneal_steps=self.sparsity_anneal_steps)
        self.masks["head"] = head_mask

    def initialize_intermediate(self):
        """Initialize MLP intermediate dimension pruning masks"""
        mask_shape = [self.base_model_info.num_hidden_layers, self.base_model_info.intermediate_size]
        num_params_per_mask = self.base_model_info.params_per_intermediate_dim
        mask_output_shape = [self.base_model_info.num_hidden_layers, 1, 1, self.base_model_info.intermediate_size] 
        
        target_int_sparsity = None
        target_mask_size = None
        if self.target_model_info is not None:
            target_int_sparsity = 1 - self.target_model_info.intermediate_size / self.base_model_info.intermediate_size
            target_mask_size = self.target_model_info.intermediate_size
            pd = {"lambda_1_intermediate": torch.nn.Parameter(torch.tensor(0.0, device=self.device)),
                  "lambda_2_intermediate": torch.nn.Parameter(torch.tensor(0.0, device=self.device))}
            self.lambdas.update(pd)
        
        int_mask = Mask(name="intermediate",
                        mask_shape=mask_shape,
                        num_params_per_mask=num_params_per_mask,
                        mask_output_shape=mask_output_shape,
                        target_sparsity=target_int_sparsity,
                        target_mask_size=target_mask_size,
                        device=self.device,
                        eval_target_model=self.eval_target_model,
                        sparsity_warmup_steps=self.sparsity_warmup_steps,
                        sparsity_anneal_steps=self.sparsity_anneal_steps)
        self.masks["intermediate"] = int_mask

    def initialize_layer(self):
        """Initialize layer pruning masks"""
        mask_shape = [self.base_model_info.num_hidden_layers]
        num_params_per_mask = self.base_model_info.params_per_layer
        mask_output_shape = [self.base_model_info.num_hidden_layers] 
        
        target_layer_sparsity = None
        target_mask_size = None
        if self.target_model_info is not None:
            target_layer_sparsity = 1 - self.target_model_info.num_hidden_layers / self.base_model_info.num_hidden_layers
            target_mask_size = self.target_model_info.num_hidden_layers
            pd = {"lambda_1_layer": torch.nn.Parameter(torch.tensor(0.0, device=self.device)),
                  "lambda_2_layer": torch.nn.Parameter(torch.tensor(0.0, device=self.device))}
            self.lambdas.update(pd)
        
        layer_mask = Mask(name="layer",
                         mask_shape=mask_shape,
                          num_params_per_mask=num_params_per_mask,
                          mask_output_shape=mask_output_shape,
                          target_sparsity=target_layer_sparsity,
                          target_mask_size=target_mask_size,
                          device=self.device,
                          eval_target_model=self.eval_target_model,
                          sparsity_warmup_steps=self.sparsity_warmup_steps,
                          sparsity_anneal_steps=self.sparsity_anneal_steps) 
        self.masks["layer"] = layer_mask

    def apply_masks_to_model(self, model, masks):
        """Apply learned masks to the actual BGE model"""
        with torch.no_grad():
            # Normalize key access (accept *_z or plain names)
            layer_key = 'layer_z' if 'layer_z' in masks else ('layer' if 'layer' in masks else None)
            if layer_key is not None:
                layer_mask = masks[layer_key]  # [num_layers]
                layers_to_keep = torch.nonzero(layer_mask > 0.5).squeeze(-1)
                
                # Remove pruned layers
                new_layers = nn.ModuleList()
                for i in layers_to_keep:
                    new_layers.append(model.encoder.layer[i])
                model.encoder.layer = new_layers
                
            # Apply head masks  
            head_key = 'head_z' if 'head_z' in masks else ('head' if 'head' in masks else None)
            if head_key is not None:
                head_mask = masks[head_key]  # [num_layers, num_heads]
                for layer_idx, layer in enumerate(model.encoder.layer):
                    if layer_idx < head_mask.size(0):
                        layer_head_mask = head_mask[layer_idx]
                        heads_to_keep = torch.nonzero(layer_head_mask > 0.5).squeeze(-1)
                        
                        # Prune attention heads
                        self._prune_attention_heads(layer.attention.self, heads_to_keep)
                        
            # Apply intermediate masks
            int_key = 'intermediate_z' if 'intermediate_z' in masks else ('intermediate' if 'intermediate' in masks else None)
            if int_key is not None:
                int_mask = masks[int_key]  # [num_layers, intermediate_size]
                for layer_idx, layer in enumerate(model.encoder.layer):
                    if layer_idx < int_mask.size(0):
                        layer_int_mask = int_mask[layer_idx]
                        dims_to_keep = torch.nonzero(layer_int_mask > 0.5).squeeze(-1)
                        
                        # Prune MLP intermediate dimensions
                        self._prune_mlp_intermediate(layer.intermediate, layer.output, dims_to_keep)

    def _prune_attention_heads(self, attention_layer, heads_to_keep):
        """Prune attention heads in a layer"""
        head_dim = attention_layer.attention_head_size
        
        # Create index tensor for keeping specific heads
        keep_indices = []
        for head_idx in heads_to_keep:
            start_idx = head_idx * head_dim
            end_idx = (head_idx + 1) * head_dim
            keep_indices.extend(range(start_idx, end_idx))
        
        keep_indices = torch.tensor(keep_indices, dtype=torch.long)
        
        # Prune Q, K, V projections
        attention_layer.query.weight.data = attention_layer.query.weight.data[keep_indices, :]
        attention_layer.query.bias.data = attention_layer.query.bias.data[keep_indices]
        
        attention_layer.key.weight.data = attention_layer.key.weight.data[keep_indices, :]
        attention_layer.key.bias.data = attention_layer.key.bias.data[keep_indices]
        
        attention_layer.value.weight.data = attention_layer.value.weight.data[keep_indices, :]
        attention_layer.value.bias.data = attention_layer.value.bias.data[keep_indices]
        
        # Prune output projection
        attention_layer.dense.weight.data = attention_layer.dense.weight.data[:, keep_indices]

    def _prune_mlp_intermediate(self, intermediate_layer, output_layer, dims_to_keep):
        """Prune MLP intermediate dimensions"""
        # Prune intermediate layer
        intermediate_layer.dense.weight.data = intermediate_layer.dense.weight.data[dims_to_keep, :]
        intermediate_layer.dense.bias.data = intermediate_layer.dense.bias.data[dims_to_keep]
        
        # Prune output layer
        output_layer.dense.weight.data = output_layer.dense.weight.data[:, dims_to_keep]

    def get_expected_num_params(self, expected_scores: dict):
        """Calculate expected parameter count based on mask scores"""
        num_parameters = 0
       
        if "head" in expected_scores:
            head_score = expected_scores["head"]  # [num_layers, num_heads]
            num_parameters += torch.sum(head_score) * self.masks["head"].num_params_per_mask
            
        if "intermediate" in expected_scores:
            int_score = expected_scores["intermediate"]  # [num_layers, intermediate_size]
            num_parameters += torch.sum(int_score) * self.masks["intermediate"].num_params_per_mask
            
        if "layer" in expected_scores:
            layer_score = expected_scores["layer"]  # [num_layers]
            num_parameters += torch.sum(layer_score) * self.masks["layer"].num_params_per_mask
            
        return num_parameters
    
    def get_target_sparsity(self, pruned_steps: int, full_sparsity: float = None):
        """Get current target sparsity with warmup schedule"""
        target_sparsity = full_sparsity or self.target_sparsity
        if getattr(self, "lagrangian_warmup_steps", 0) > 0:
            target_sparsity = (target_sparsity - self.start_sparsity) * min(1, pruned_steps / self.lagrangian_warmup_steps) + self.start_sparsity
        return target_sparsity

    def lagrangian_regularization(self, pruned_steps: int):
        """Lagrangian regularization enforcing exact counts (layers, per-layer intermediates)."""
        def lag_term(cur, tgt, l1, l2):
            diff = cur - tgt
            return l1 * diff + l2 * diff * diff

        total_loss = 0.0
        metrics = {}

        # If no explicit target model: fall back to global sparsity control
        if self.target_model_info is None:
            expected_scores, _ = self.calculate_expected_score_sparsity()
            expected_size = self.get_expected_num_params(expected_scores)
            target_sparsity = self.get_target_sparsity(pruned_steps, self.target_sparsity)
            expected_sparsity = 1 - expected_size / self.prunable_model_size
            total_loss = lag_term(expected_sparsity, target_sparsity, self.lambdas["lambda_1"], self.lambdas["lambda_2"])
            metrics.update({"expected_sparsity": float(expected_sparsity), "target_sparsity": float(target_sparsity)})
        else:
            for name in self.pruning_modules:
                mask = self.masks[name]
                if mask.target_mask_size is None:
                    continue
                expected_active = mask.expected_active()  # tensor (per-layer) or scalar
                target = mask.target_mask_size
                if expected_active.dim() == 0:
                    lt = lag_term(expected_active, target, self.lambdas[f"lambda_1_{name}"], self.lambdas[f"lambda_2_{name}"])
                    metrics[f"active_{name}"] = float(expected_active.item())
                    metrics[f"target_{name}"] = float(target)
                else:
                    # per-layer intermediates
                    lt = lag_term(expected_active, target, self.lambdas[f"lambda_1_{name}"], self.lambdas[f"lambda_2_{name}"])
                    metrics[f"active_{name}_mean"] = float(expected_active.mean().item())
                    metrics[f"target_{name}"] = float(target)
                total_loss = total_loss + lt.mean()

        # Optional entropy regularization to push masks toward binary states
        entropy_reg = 0.0
        for m in self.masks.values():
            sm = torch.sigmoid(m.z_loga / m.temperature * m.magical_number)
            entropy = -(sm * torch.log(sm + 1e-8) + (1 - sm) * torch.log(1 - sm + 1e-8))
            entropy_reg += entropy.mean()
        total_loss = total_loss - 0.01 * entropy_reg
        return total_loss, metrics

    def constrain_parameters(self):
        """Constrain mask parameters"""
        for key in self.masks:
            self.masks[key].constrain_parameters()

    def calculate_expected_score_sparsity(self):
        """Calculate expected scores and sparsities"""
        expected_scores = {}
        expected_sparsitys = {}
        for key in self.masks:
            score, sparsity = self.masks[key].calculate_expected_score_sparsity()
            expected_scores[key] = score
            expected_sparsitys[key] = sparsity
        return expected_scores, expected_sparsitys
 
    def forward(self, calculate_lagrangian: bool = False, pruned_steps: int = 0):
        """Forward pass"""
        self.constrain_parameters()
        self.current_step = pruned_steps  # Update internal step counter
        
        if calculate_lagrangian:
            return self.lagrangian_regularization(pruned_steps)
        
        zs = {f"{pruning_module}_z": [] for pruning_module in self.pruning_modules}
        
        for pruning_module in self.pruning_modules:
            mask = self.masks[pruning_module]
            z = mask.sample_z(step=pruned_steps)
            zs[f"{pruning_module}_z"] = z
            
        return zs
    
    def get_soft_sparsities(self):
        """Get soft sparsities for training metrics"""
        sparsities = {}
        for key, mask in self.masks.items():
            soft_mask = mask.get_soft_mask()
            sparsity = (1.0 - soft_mask).mean().item()
            sparsities[key] = sparsity
        return sparsities
    
    def get_export_masks(self):
        """Get deterministic masks for export"""
        zs = {}
        for pruning_module in self.pruning_modules:
            mask = self.masks[pruning_module]
            z = mask.sample_z_for_export()
            zs[f"{pruning_module}_z"] = z
        return zs

    # ---------------- Finalization (Top-K Selection) -----------------
    @torch.no_grad()
    def build_final_masks(self):
        """Construct hard masks by selecting top-k units per target counts.
        Uses current soft probabilities; enforces target_model_info if present.
        """
        final_masks = {}
        for name, mask in self.masks.items():
            sm = mask.get_soft_mask().clone()
            if name == 'layer' and mask.target_mask_size:
                k = mask.target_mask_size
                topk = torch.topk(sm.view(-1), k=k, largest=True).indices
                hard = torch.zeros_like(sm)
                hard[topk] = 1
            elif name == 'intermediate' and mask.target_mask_size:
                # per-layer uniform k
                k = mask.target_mask_size
                hard = torch.zeros_like(sm)
                for i in range(sm.size(0)):
                    vals, idx = torch.topk(sm[i], k=k, largest=True)
                    hard[i, idx] = 1
            else:
                hard = (sm > 0.5).float()
            final_masks[f"{name}_z"] = hard.reshape(mask.mask_output_shape)
        return final_masks

    @torch.no_grad()
    def finalize_architecture(self, backbone):
        """Apply structural pruning based on top-k masks and drop gating overhead."""
        fm = self.build_final_masks()
        self.apply_masks_to_model(backbone, fm)
        return fm

def create_pruned_bge_model(original_model, l0_module, final_masks):
    """Create final pruned BGE model"""
    import copy
    pruned_model = copy.deepcopy(original_model)
    
    # Apply masks to create the final pruned architecture
    l0_module.apply_masks_to_model(pruned_model, final_masks)
    
    return pruned_model

# Usage example:
def test_bge_l0_module():
    """Test the BGE L0 module"""
    from omegaconf import OmegaConf as om 
    from transformers import AutoModel
    
    # Load BGE model
    bge_model = AutoModel.from_pretrained("BAAI/bge-m3")
    
    # Create config
    cfg = om.create({
        "l0_module": {
            "pruning_modules": ["layer", "head", "intermediate"],
            "target_sparsity": 0.5,
            "start_sparsity": 0.0,
            "lagrangian_warmup_steps": "1000ba",
            "sparsity_warmup_steps": "500ba",
            "sparsity_anneal_steps": "2000ba",
            "eval_target_model": True
        }
    })
    
    l0_module = L0ModuleEmbedding(cfg, "cpu", bge_model)
    
    # Test forward pass
    l0_module.train()
    zs = l0_module.forward()
    for key in zs:
        print(key, zs[key].shape if hasattr(zs[key], 'shape') else zs[key])

    # Test lagrangian loss
    loss, metrics = l0_module(calculate_lagrangian=True, pruned_steps=500)
    print("Lagrangian loss:", loss.item())
    for key, val in metrics.items():
        print(f"{key}: {val}")

if __name__ == "__main__":
    test_bge_l0_module()
