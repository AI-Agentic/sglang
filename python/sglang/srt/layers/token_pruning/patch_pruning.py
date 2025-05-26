from typing import Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiversityPatchPruning:

    """
    Select a diverse subset of ViT patch embeddings using farthest‑point sampling
    in cosine‑similarity space.

    Parameters
    ----------
    vit_embeds : torch.Tensor
        Patch embeddings shaped (B, N, C) where
        B = #patches in the batch dimension to prune from,
        N = tokens per patch (e.g. sequence length in ViT),
        C = hidden size / embedding dimension.
    pruning_ratio : float
        Fraction of patches to *keep* in (0, 1].  A value ≤ 0 raises.
    verbose : bool, optional
        If True, prints basic debug statistics.

    Returns
    -------
    torch.Tensor
        Tensor shaped (⌈B·pruning_ratio⌉, N, C) containing only the
        selected (kept) patches.  Order is deterministic.
    """
    def __init__(self, 
                 pruning_ratio:float, 
                 verbose: Optional[bool] = False):
        super().__init__()
        if not (0.0 < pruning_ratio <= 1.0):
            raise ValueError("pruning_ratio must be in (0, 1].")
        self.pruning_ratio=pruning_ratio
        self.verbose = verbose

    def forward(self, vit_embeds:torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if vit_embeds.ndim != 3:
            raise ValueError(
                f"vit_embeds should have shape (B, N, C); got {vit_embeds.shape}"
            )

        B, N, C = vit_embeds.shape
        device = vit_embeds.device
        k = max(1, int(int(B) * (1 - self.pruning_ratio)))

        if self.verbose:
            print(f"[Patch Pruning] patches_in={B}  patches_out={k}")
            print("[Patch Pruning] B,N,C: ", B, N, C)
            print("[Patch Pruning] k,N,C: ", k, N, C)


        # No pruning needed
        if k >= B:
            return vit_embeds

        # ------------------------------------------------------------------
        #  Cosine farthest‑point sampling (FPS)
        # ------------------------------------------------------------------
        #   1. Flatten per‑patch tensors to (B, N*C).
        #   2. Normalize so cosine similarity == dot product.
        #   3. Repeatedly pick the patch least similar to anything already chosen.
        #
        #   Complexity: O(B·k) space‑efficient (no B×B matrix held in memory).
        # ------------------------------------------------------------------
        flat = F.normalize(vit_embeds.view(B, -1), dim=-1)  # (B, N*C)

        # idx_selected will end up length k
        idx_selected = torch.empty(k, dtype=torch.long, device=device)
        # 1️⃣  seed with the most “central” patch (highest L2 norm == largest dot w/ itself),
        #     equivalent to random but deterministic and data‑driven.
        idx_selected[0] = flat.pow(2).sum(1).argmax()

        # Keep track of each patch’s *best* similarity to anything already chosen.
        # Initialise with +∞ so the first update always wins.
        best_sim = torch.full((B,), float("inf"), device=device)

        for i in range(1, k):
            # Cosine similarity between all patches and the last selected one
            sim = torch.matmul(flat, flat[idx_selected[i - 1]].unsqueeze(0).T).squeeze(1)
            best_sim = torch.minimum(best_sim, sim)  # elementwise min (lower ⇒ more diverse)

            # Never pick an already‑selected index again
            best_sim[idx_selected[:i]] = float("inf")

            # Next patch = one with *lowest* best_sim so far (most dissimilar)
            idx_selected[i] = best_sim.argmin()

        pruned = vit_embeds[idx_selected]  # (k, N, C)

        return pruned
