from sglang.srt.layers.token_pruning.utils import DoNothing, nearest_square
from sglang.srt.layers.token_pruning.patch_pruning import DiversityPatchPruning
from sglang.srt.layers.token_pruning.visionzip import VisionZipTokenPruning
from sglang.srt.layers.token_pruning.big_attn import BigAttnTokenPruning

TOKEN_LEVEL_PRUNING_ALG = [
    "tome",
    "visionzip",
    "big-attn"
]

PATCH_LEVEL_PRUNING_ALG = [
    "patch-pruning"
]

MIXED_PRUNING_ALG = []


VLM_TOKEN_PRUNING_SUPPORTED_DICT = {
    "InternVLChatModel" : ["patch-pruning", "visionzip", "big-attn"],
}

TOKEN_PRUNIGN_NEED_METRIC_ALG = ["visionzip", "big-attn"]