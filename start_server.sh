
# export TORCHINDUCTOR_CACHE_DIR=~/.triton
# CUDA_VISIBLE_DEVICES=4 python3 -m sglang.launch_server \
#     --model-path OpenGVLab/InternVL3-1B \
#     --trust-remote-code \
#     --chat-template internvl-2-5 \
#     --mem-fraction-static 0.2 \
#     --token-pruning-alg tome \
#     --token-pruning-ratio 0.5



# export TORCHINDUCTOR_CACHE_DIR=~/.triton
# CUDA_VISIBLE_DEVICES=4 python3 -m sglang.launch_server \
#     --model-path Qwen/Qwen2.5-VL-3B-Instruct \
#     --trust-remote-code \
#     --chat-template qwen2-vl \
#     --mem-fraction-static 0.2 \
#     --token-pruning-alg tome \
#     --token-pruning-ratio 0.5


# rm -rf ~/.cache/flashinfer/
# rm -r ~/.triton/*

python3 -m sglang_router.launch_server \
    --model-path OpenGVLab/InternVL3-8B \
    --dp-size 8 \
    --trust-remote-code \
    --chat-template internvl-2-5 \
    --mem-fraction-static 0.5 \
    --chunked-prefill-size 2048 \
    --max-running-requests 10 \
    --router-balance-rel-threshold 1.1 \
    --router-balance-abs-threshold 1 \
    --router-max-payload-size 10485760 \
    --host 0.0.0.0 \
    --port 23333
    # --token-pruning-alg patch-pruning \
    # --token-pruning-ratio 0.3 



    # --token-pruning-alg patch-pruning \
    # --token-pruning-ratio 0.3 \