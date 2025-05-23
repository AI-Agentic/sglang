
export TORCHINDUCTOR_CACHE_DIR=~/.triton
CUDA_VISIBLE_DEVICES=4 python3 -m sglang.launch_server \
    --model-path OpenGVLab/InternVL3-1B \
    --trust-remote-code \
    --chat-template internvl-2-5 \
    --mem-fraction-static 0.2 \
    --token-pruning-alg tome \
    --token-pruning-ratio 0.5



# export TORCHINDUCTOR_CACHE_DIR=~/.triton
# CUDA_VISIBLE_DEVICES=4 python3 -m sglang.launch_server \
#     --model-path Qwen/Qwen2.5-VL-3B-Instruct \
#     --trust-remote-code \
#     --chat-template qwen2-vl \
#     --mem-fraction-static 0.2 \
#     --token-pruning-alg tome \
#     --token-pruning-ratio 0.5