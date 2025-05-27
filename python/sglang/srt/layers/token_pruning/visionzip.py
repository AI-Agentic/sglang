from typing import Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.srt.layers.token_pruning.utils import nearest_square


class VisionZipTokenPruning:
    '''
    VisionZip: Longer is Better but Not Necessary in Vision Language Models
                https://arxiv.org/pdf/2412.04467
    
    Intro: Token pruning is divided into two steps. 
            First, the key token is obtained by the attention value of other tokens and cls token, 
            and then the remaining tokens are compressed together.
    '''
    def __init__(self, 
                 pruning_ratio:float, 
                 dominant_token_ratio: Optional[float] = 0.85, 
                 downsample_ratio: Optional[float] = 1.0,
                 verbose: Optional[bool] = False):
        super().__init__()
        if not (0.0 < pruning_ratio <= 1.0):
            raise ValueError("pruning_ratio must be in (0, 1].")
        if not (0.0 < dominant_token_ratio <= 1.0):
            raise ValueError("dominant_token_ratio must be in (0, 1].")
        
        self.pruning_ratio=pruning_ratio
        self.downsample_ratio = downsample_ratio 
        self.dominant_token_ratio = dominant_token_ratio # token taked by attention value with cls token
        # contextual_token_ratio = 1 - dominant_token_ratio , as default.
        self.verbose = verbose

    def cal_tokens_to_prune(self, original_token: int) -> Tuple[int, int]:
        """
        Calculate the number of tokens to prune based on the original number of tokens, pruning ratio and downsample_ratio.
        
        Parameters:
        ----------
        original_tokens : int
            The original number of tokens before pruning.
        
        Returns:
        -------
        (dominant_tokens, contextual_tokens) : Tuple[int, int]
        """
        token_nums_out_original = int(original_token * (self.downsample_ratio ** 2)) # after pixel_shuffle
        token_nums_out_after_pruning = int(token_nums_out_original * (1 - self.pruning_ratio))
        token_nums_out_after_pruning = nearest_square(token_nums_out_after_pruning) # ensure the number of tokens is a perfect square

        token_nums_vit_after_pruning = int(token_nums_out_after_pruning / (self.downsample_ratio ** 2)) # before pixel_shuffle

        dominant_token = int(token_nums_vit_after_pruning * self.dominant_token_ratio)
        contextual_token = token_nums_vit_after_pruning - dominant_token
        if self.verbose:
            corrected_pruning_ratio = 1 - (token_nums_vit_after_pruning / original_token)
            print(f"[Token Pruning] Original tokens in ViT: {original_token}, Original tokens after downsample: {token_nums_out_original}, Tokens after downsample and pruning: {token_nums_out_after_pruning}")
            print(f"[Token Pruning] Dominant tokens: {dominant_token}, Contextual tokens: {contextual_token}")
            print(f"[Token Pruning] In order for the output to be exactly squared, change pruning ratio to: {corrected_pruning_ratio:.2f}")
        return dominant_token, contextual_token
        
    
    def forward(self, 
                vit_embeds:torch.Tensor,
                metric: Tuple[torch.Tensor, torch.Tensor]
                ) -> torch.Tensor:  
        # metric: (attention_value, key_value)

        if vit_embeds.ndim != 3:
            raise ValueError(
                f"vit_embeds should have shape (B, N, C); got {vit_embeds.shape}"
            )
        original_token_num = int(vit_embeds.shape[1])
        dominant_num, contextual_num = self.cal_tokens_to_prune(original_token_num)
        attn_weight, key_metric = metric
        key_metric = key_metric[:, 1:, :]  # Exclude cls token

        # 1. Get dominant tokens based on cls attention value
        cls_attention = attn_weight[:, :, 0, 1:].sum(dim=1) # (B, N-1)
        topk_indices = cls_attention.topk(dominant_num, dim=1).indices
        mask = torch.ones_like(vit_embeds[:, :, 0], dtype=torch.bool, device=vit_embeds.device).scatter_(1, topk_indices, False).unsqueeze(-1)
        dominant_tokens = vit_embeds.masked_select(~mask).view(vit_embeds.shape[0], dominant_num, vit_embeds.shape[2])

        # 2. Filter out dominant tokens
        metric_filtered = key_metric.masked_select(mask).view(key_metric.shape[0], key_metric.shape[1] - dominant_num, key_metric.shape[2])
        vit_embeds_filtered = vit_embeds.masked_select(mask).view(vit_embeds.shape[0], vit_embeds.shape[1] - dominant_num, vit_embeds.shape[2])  
        metric_normalized = metric_filtered / metric_filtered.norm(dim=-1, keepdim=True) 

        # 3. Get contextual tokens based on similarity (same as ToMe)
        step = max(1, metric_normalized.shape[1] // contextual_num)
        target_indices = torch.arange(0, metric_normalized.shape[1], step, device=metric_normalized.device)[:contextual_num]
        target_tokens = metric_normalized[:, target_indices, :]

        tokens_to_merge = metric_normalized[:, ~torch.isin(torch.arange(metric_normalized.shape[1], device=metric_normalized.device), target_indices), :]
        similarity = torch.bmm(tokens_to_merge, target_tokens.transpose(1, 2))
        assign_one_hot = torch.zeros(tokens_to_merge.shape[0], tokens_to_merge.shape[1], contextual_num, dtype=vit_embeds_filtered.dtype, device=metric_normalized.device)
        assign_one_hot.scatter_(2, similarity.argmax(dim=2).unsqueeze(-1), 1)
        counts = assign_one_hot.sum(dim=1).clamp(min=1).unsqueeze(-1)
        hidden_to_merge = vit_embeds_filtered[:, ~torch.isin(torch.arange(vit_embeds_filtered.shape[1], device=vit_embeds_filtered.device), target_indices), :]
        aggregated_hidden = torch.bmm(assign_one_hot.transpose(1, 2), hidden_to_merge) / counts
        target_hidden = vit_embeds_filtered[:, target_indices, :]  
        
        contextual_tokens = target_hidden + aggregated_hidden

        # 4. Merge with target hidden states and concatenate
        pruned_vit_embeds = torch.cat([dominant_tokens, contextual_tokens], dim=1)

        if self.verbose:
            print(f"[Token Pruning] Dominant tokens shape: {dominant_tokens.shape}, Contextual tokens shape: {contextual_tokens.shape}")

        return pruned_vit_embeds