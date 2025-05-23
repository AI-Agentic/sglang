from sglang.srt.layers.token_pruning.utils import DoNothing
from sglang.srt.layers.token_pruning.patch_pruning import DiversityPatchPruning


TOKEN_PRUNING_SUPPORTED_MODELS = [
    "InternVLChatModel"
]

TOKEN_LEVEL_PRUNING_ALG = [
    "tome",
    "visionzip"
]

PATCH_LEVEL_PRUNING_ALG = [
    "patch-pruning"
]

MIXED_PRUNING_ALG = []