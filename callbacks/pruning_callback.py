import torch
from composer.core import Callback, State
from composer.loggers import Logger
from typing import Dict, Any

class PruningCallback(Callback):
    """Minimal callback for pruning monitoring"""
    
    def __init__(self, log_interval: int = 500):
        self.log_interval = log_interval
        self.step_count = 0
    
    def batch_end(self, state: State, logger: Logger) -> None:
        """Called at the end of each training batch"""
        self.step_count += 1
        
        # Sync model's step counter with Composer state
        model = state.model
        if hasattr(model, 'sync_step_with_composer_state'):
            model.sync_step_with_composer_state(state)
        
        if self.step_count % self.log_interval == 0:
            self._log_pruning_metrics(state, logger)
            self._update_lagrangian_multipliers(state)
    
    def _log_pruning_metrics(self, state: State, logger: Logger) -> None:
        """Log pruning-specific metrics"""
        model = state.model
        
        if not hasattr(model, 'l0_module'):
            return
            
        l0_module = model.l0_module
        
        # Constrain parameters to valid ranges
        l0_module.constrain_parameters()
        
        metrics = {}
        
        # Get current step for calculations
        current_step = state.timestamp.batch.value if hasattr(state.timestamp, 'batch') else 0
        
        # Log sparsity metrics - use soft sparsities during training
        try:
            if state.model.training:
                # During training: use soft sparsities
                soft_sparsities = l0_module.get_soft_sparsities()
                for mask_name, sparsity in soft_sparsities.items():
                    metrics[f'sparsity/{mask_name}'] = sparsity
            else:
                # During eval: use hard sparsities
                expected_scores, expected_sparsitys = l0_module.calculate_expected_score_sparsity()
                for mask_name, sparsity in expected_sparsitys.items():
                    sparsity_val = sparsity.mean().item() if hasattr(sparsity, 'mean') else sparsity
                    metrics[f'sparsity/{mask_name}'] = sparsity_val
        except:
            # Fallback: try to get sparsity from forward pass
            try:
                zs = l0_module(pruned_steps=current_step)
                for mask_name, mask_tensor in zs.items():
                    base_name = mask_name.replace('_z', '')
                    if state.model.training:
                        # Soft sparsity for training
                        sparsity = (1.0 - mask_tensor).mean().item()
                    else:
                        # Hard sparsity for eval
                        sparsity = (mask_tensor == 0).float().mean().item()
                    metrics[f'sparsity/{base_name}'] = sparsity
            except:
                pass
        
        # Log to console
        if metrics:
            sparsity_str = ", ".join([f"{k.split('/')[-1]}: {v:.3f}" for k, v in metrics.items()])
            print(f"Step {self.step_count}: Sparsity - {sparsity_str}")
        
        # Log to logger if available
        if logger and hasattr(logger, 'log_metrics') and metrics:
            logger.log_metrics(metrics)
        
        if hasattr(model, 'l0_module'):
            l0_module = model.l0_module
            
            # Log constraint violations if available
            try:
                constraint_loss = self._compute_constraint_violations(l0_module)
                logger.log_metrics({"train_constraint_loss": constraint_loss})
            except:
                pass
            
            # Log effective model size if available  
            try:
                effective_params = self._compute_effective_params(l0_module)
                logger.log_metrics({"train_effective_params": effective_params})
            except:
                pass
    
    def _update_lagrangian_multipliers(self, state: State) -> None:
        """Update Lagrangian multipliers based on constraint violations"""
        # The L0Module handles its own Lagrangian multiplier updates
        # This is now a no-op as the updates happen in the forward pass
        pass
    
    def _adjust_lagrangian_multipliers(self, l0_module) -> None:
        """Adjust Lagrangian multipliers based on constraint satisfaction"""
        # This functionality is handled by the L0Module itself
        pass
    
    def _compute_constraint_violations(self, l0_module) -> float:
        """Compute total constraint violations"""
        total_violation = 0.0
        
        try:
            expected_scores, expected_sparsitys = l0_module.calculate_expected_score_sparsity()
            for mask_name, sparsity in expected_sparsitys.items():
                if hasattr(l0_module.masks[mask_name], 'target_sparsity'):
                    target = l0_module.masks[mask_name].target_sparsity
                    if target is not None:
                        current = sparsity.mean().item() if hasattr(sparsity, 'mean') else sparsity
                        violation = abs(current - target)
                        total_violation += violation
        except:
            pass
        
        return total_violation
    
    def _compute_effective_params(self, l0_module) -> int:
        """Compute effective number of parameters after pruning"""
        total_effective = 0
        
        try:
            expected_scores, _ = l0_module.calculate_expected_score_sparsity()
            for mask_name, score in expected_scores.items():
                if hasattr(l0_module.masks[mask_name], 'num_params_per_mask'):
                    effective_size = score.sum().item()
                    total_effective += int(effective_size * l0_module.masks[mask_name].num_params_per_mask)
        except:
            pass
        
        return total_effective
    
    def _compute_mask_statistics(self, l0_module) -> Dict[str, float]:
        """Compute statistics about mask distributions"""
        stats = {}
        
        try:
            for mask_name, mask in l0_module.masks.items():
                if hasattr(mask, 'z_loga'):
                    z_loga = mask.z_loga
                    
                    # Mean and std of log-alpha values
                    stats[f"{mask_name}_mean_log_alpha"] = z_loga.mean().item()
                    stats[f"{mask_name}_std_log_alpha"] = z_loga.std().item()
                    
                    # Expected sparsity
                    if hasattr(mask, 'calculate_expected_score_sparsity'):
                        _, sparsity = mask.calculate_expected_score_sparsity()
                        stats[f"{mask_name}_expected_sparsity"] = sparsity.mean().item()
        except:
            pass
        
        return stats
    
    def _log_mask_visualizations(self, state: State, logger: Logger) -> None:
        """Create and log mask visualizations"""
        model = state.model
        
        if hasattr(model, 'l0_module'):
            l0_module = model.l0_module
            
            try:
                for mask_name, mask in l0_module.masks.items():
                    # Create histogram of mask values
                    try:
                        import matplotlib.pyplot as plt
                        fig, ax = plt.subplots(figsize=(8, 6))
                        
                        if hasattr(mask, 'z_loga'):
                            z_loga = mask.z_loga.detach().cpu().numpy()
                            if z_loga.ndim > 1:
                                z_loga = z_loga.flatten()
                            
                            ax.hist(z_loga, bins=50, alpha=0.7, edgecolor='black')
                            ax.set_xlabel('Log-alpha values')
                            ax.set_ylabel('Frequency')
                            ax.set_title(f'Distribution of {mask_name} mask values')
                            ax.grid(True, alpha=0.3)
                            
                            # Log the figure
                            if hasattr(logger, 'log_images'):
                                logger.log_images({f"mask_distribution_{mask_name}": fig})
                            plt.close(fig)
                    except (ImportError, Exception):
                        pass  # Skip visualization if matplotlib not available or any other error
            except:
                pass
    
    def epoch_end(self, state: State, logger: Logger) -> None:
        """Called at the end of each epoch"""
        self._log_pruning_summary(state, logger)
    
    def _log_pruning_summary(self, state: State, logger: Logger) -> None:
        """Log summary of pruning progress"""
        model = state.model
        
        if hasattr(model, 'l0_module'):
            l0_module = model.l0_module
            
            # Log target vs current architecture
            if hasattr(l0_module, 'target_model_info') and l0_module.target_model_info is not None:
                target_info = l0_module.target_model_info
                base_info = l0_module.base_model_info
                
                reduction_stats = {
                    'target_hidden_reduction': 1 - target_info.hidden_size / base_info.hidden_size,
                    'target_layer_reduction': 1 - target_info.num_hidden_layers / base_info.num_hidden_layers,
                    'target_head_reduction': 1 - target_info.num_attention_heads / base_info.num_attention_heads,
                    'target_intermediate_reduction': 1 - target_info.intermediate_size / base_info.intermediate_size
                }
                
                for key, value in reduction_stats.items():
                    logger.log_metrics({f"pruning_{key}": value})