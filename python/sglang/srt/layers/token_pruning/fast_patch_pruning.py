from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class FastDiversityPatchPruning(nn.Module):
    """
    Ultra-fast version using pure tensor operations.
    No loops, no sampling - just matrix operations.
    """
    
    def __init__(self, 
                 pruning_ratio: float,
                 diversity_weight: float = 1.0,
                 verbose: Optional[bool] = False):
        super().__init__()
        if not (0.0 < pruning_ratio <= 1.0):
            raise ValueError("pruning_ratio must be in (0, 1].")
        
        self.pruning_ratio = pruning_ratio
        self.keep_ratio = 1 - pruning_ratio
        self.diversity_weight = diversity_weight
        self.verbose = verbose
        
    def forward(self, 
                vit_embeds:torch.Tensor,
                metric: Tuple[torch.Tensor, torch.Tensor]
                ) -> torch.Tensor: 
        if vit_embeds.ndim != 3:
            raise ValueError(f"Expected shape (B, N, C), got {vit_embeds.shape}")

        B, N, C = vit_embeds.shape
        k = max(1, int(B * self.keep_ratio))
        
        if self.verbose:
            print(f"[Ultra-Fast Pruning] patches_in={B}, patches_out={k}")

        if k >= B:
            return vit_embeds

        # Normalize embeddings
        flat = F.normalize(vit_embeds.view(B, -1), dim=-1)
        
        # Compute importance scores combining magnitude and diversity
        importance = self._compute_importance_scores(flat)
        
        # Select top-k most important patches
        _, top_indices = torch.topk(importance, k, dim=0)
        
        return vit_embeds[top_indices]
    
    def _compute_importance_scores(self, flat_embeds: torch.Tensor) -> torch.Tensor:
        """
        Compute importance scores balancing individual strength and diversity.
        Pure tensor operations - no loops!
        """
        B = flat_embeds.shape[0]
        device = flat_embeds.device
        
        # Individual importance (L2 norm of embeddings)
        individual_importance = flat_embeds.norm(dim=1)
        
        # Diversity importance: how different each patch is from the mean
        mean_embed = flat_embeds.mean(dim=0, keepdim=True)
        diversity_scores = 1.0 - F.cosine_similarity(flat_embeds, mean_embed.expand(B, -1))
        
        # Combine scores
        total_importance = individual_importance + self.diversity_weight * diversity_scores
        
        return total_importance


