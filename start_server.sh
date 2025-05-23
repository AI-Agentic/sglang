
export TORCHINDUCTOR_CACHE_DIR=~/.triton
CUDA_VISIBLE_DEVICES=7 python3 -m sglang.launch_server \
    --model-path OpenGVLab/InternVL3-1B \
    --trust-remote-code \
    --chat-template internvl-2-5 \
    --mem-fraction-static 0.2 \
    --token-pruning-alg patch-pruning \
    --token-pruning-ratio 0.5 \
    --disable-radix-cache
