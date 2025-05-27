
# export TORCHINDUCTOR_CACHE_DIR=~/.triton
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server \
    --model-path OpenGVLab/InternVL3-38B \
    --chat-template internvl-2-5 \
    --mem-fraction-static 0.5 \
    --chunked-prefill-size 2048 \
    --max-running-requests 10 \
    --token-pruning-alg patch-pruning \
    --token-pruning-ratio 0.5 \
    --debug-token-pruning \
    --trust-remote-code \
    --disable-radix-cache \
    --port 30000 \
    --dp 4

# CUDA_VISIBLE_DEVICES=1,2,3,4 python3 -m sglang_router.launch_server \
#     --model-path OpenGVLab/InternVL3-8B \
#     --dp-size 4 \
#     --chat-template internvl-2-5 \
#     --mem-fraction-static 0.5 \
#     --token-pruning-alg patch-pruning \
#     --token-pruning-ratio 0.5 \
#     --port 23333 

# export TORCHINDUCTOR_CACHE_DIR=~/.triton
# CUDA_VISIBLE_DEVICES=4 python3 -m sglang.launch_server \
#     --model-path Qwen/Qwen2.5-VL-3B-Instruct \
#     --chat-template qwen2-vl \
#     --mem-fraction-static 0.5 


# python3 -m sglang_router.launch_server \
#     --model-path OpenGVLab/InternVL3-8B \
#     --data-parallel-size 8 \
#     --chat-template internvl-2-5 \
#     --mem-fraction-static 0.7 \
#     --port 23333
    # --token-pruning-alg patch-pruning \
    # --token-pruning-ratio 0.5


# python3 -m sglang_router.launch_server \
#     --model-path OpenGVLab/InternVL3-8B \
#     --dp-size 8 \
#     --chat-template internvl-2-5 \
#     --mem-fraction-static 0.5 \
#     --chunked-prefill-size 2048 \
#     --max-running-requests 10 \
#     --router-balance-rel-threshold 1.1 \
#     --router-balance-abs-threshold 1 \
#     --router-max-payload-size 10485760 \
#     --token-pruning-alg patch-pruning \
#     --token-pruning-ratio 0.1 \
#     --disable-radix-cache