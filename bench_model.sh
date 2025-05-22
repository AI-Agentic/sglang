
export TORCHINDUCTOR_CACHE_DIR=~/.triton
CUDA_VISIBLE_DEVICES=0 python -m sglang.bench_one_batch \
    --model-path OpenGVLab/InternVL3-8B \
    --chat-template internvl-2-5 \
    --mem-fraction-static 0.5 \
    --port 30000 \
    --run-name "bench_test" \
    --batch-size 1 \
    --input-len 8192 \
    --output-len 32 \
    --result-filename "bench_test"