from typing import Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.srt.layers.token_pruning.utils import nearest_square


class BigAttnTokenPruning:
    '''
    Intro: Token pruning based on attention weight.
            torch.norm() and leave the top-k biggest tokens.
    '''
    def __init__(self, 
                 pruning_ratio:float, 
                 downsample_ratio: Optional[float] = 1.0,
                 verbose: Optional[bool] = False):
        super().__init__()
        if not (0.0 < pruning_ratio <= 1.0):
            raise ValueError("pruning_ratio must be in (0, 1].")
        
        self.pruning_ratio=pruning_ratio
        self.downsample_ratio = downsample_ratio 
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

        if self.verbose:
            corrected_pruning_ratio = 1 - (token_nums_vit_after_pruning / original_token)
            print(f"[Token Pruning] Original tokens in ViT: {original_token}, Original tokens after downsample: {token_nums_out_original}, Tokens after downsample and pruning: {token_nums_out_after_pruning}")
            print(f"[Token Pruning] In order for the output to be exactly squared, change pruning ratio to: {corrected_pruning_ratio:.2f}")
        return token_nums_vit_after_pruning
        
    
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
        token_nums_vit_after_pruning = self.cal_tokens_to_prune(original_token_num)
        attn_weight, key_metric = metric

        # cls_attention = attn_weight[:, :, 0, 1:].sum(dim=1) # (B, N-1)
        attn_no_cls = attn_weight[:, :, 1:, 1:]  # Exclude cls token
        attn_norm = attn_no_cls.norm(dim=2).sum(dim=1)
        topk_indices = attn_norm.topk(token_nums_vit_after_pruning, dim=1).indices
        mask = torch.ones_like(vit_embeds[:, :, 0], dtype=torch.bool, device=vit_embeds.device).scatter_(1, topk_indices, False).unsqueeze(-1)
        dominant_tokens = vit_embeds.masked_select(~mask).view(vit_embeds.shape[0], token_nums_vit_after_pruning, vit_embeds.shape[2])

        if self.verbose:
            print(f"[Token Pruning]  Shape after pruning: {dominant_tokens.shape}")

        return dominant_tokens