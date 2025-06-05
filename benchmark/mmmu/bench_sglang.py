"""
Bench the sglang-hosted vLM with benchmark MMMU

Usage:
    Host the VLM: python -m sglang.launch_server --model-path Qwen/Qwen2-VL-7B-Instruct --port 30000

    Benchmark: python benchmark/mmmu/bench_sglang.py --port 30000 --concurrency 16

The eval output will be logged
"""

import argparse
import asyncio
import sys
import time
import base64
import traceback
from dataclasses import dataclass, field
from typing import Any, List, Tuple

import cv2
import numpy as np
import aiohttp
import openai
from data_utils import save_json
from eval_utils import (
    EvalArgs,
    eval_result,
    get_sampling_params,
    prepare_samples,
    process_result,
)
from tqdm import tqdm

from sglang.test.test_utils import add_common_sglang_args_and_parse

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=20 * 60 * 60)


@dataclass
class RequestFuncOutput:
    generated_text: List[str] = field(default_factory=list)
    prompt_len: List[int] = field(default_factory=list)
    output_len: List[int] = field(default_factory=list)
    latency: List[float] = field(default_factory=list)
    ttft: List[float] = field(default_factory=list)
    itl: List[float] = field(default_factory=list)  # List of inter-token latencies

    success: bool = False
    error: str = ""


async def async_request_profile(api_url: str) -> RequestFuncOutput:
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        output = RequestFuncOutput()
        try:
            async with session.post(url=api_url) as response:
                if response.status == 200:
                    output.success = True
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    return output


def _get_prefix_suffix(prompt: str) -> Tuple[str, str]:
    """Split the prompt into prefix and suffix."""
    prefix = prompt.split("<")[0]
    suffix = prompt.split(">", 1)[1]
    return prefix, suffix


async def process_sample(
    client: Any, sample: dict, sampling_params: dict
) -> Tuple[dict, str]:
    """Send a single sample to the LLM and return (sample, response)."""
    prompt = sample["final_input_prompt"]
    prefix, suffix = _get_prefix_suffix(prompt)
    image = sample["image"]
    assert image is not None
    image_path = sample["image_path"]


    def _pad_to_nearest_patch_size(img: np.ndarray, patch_size = 448) -> np.ndarray:
        h, w = img.shape[:2]
        
        # 计算新的宽度和高度（向上取整为 patch_size 的倍数）
        new_w = ((w + patch_size - 1) // patch_size) * patch_size
        new_h = ((h + patch_size - 1) // patch_size) * patch_size
        # 计算需要在每侧填充的像素数
        top = 0 
        bottom = (new_h - h)
        left = 0
        right = (new_w - w)
        
        # 改为使用纯白色填充
        border_value = [255, 255, 255]  # 纯白色
        
        # 进行填充操作
        padded_img = cv2.copyMakeBorder(
            img,
            top=top,
            bottom=bottom,
            left=left,
            right=right,
            borderType=cv2.BORDER_CONSTANT,
            value=border_value
        )
        
        return padded_img

    def concat_images(images: List[np.ndarray], height_concat = True) -> np.ndarray:

        """
        Concatenate a list of images either vertically or horizontally.
        
        :param images: List of images (numpy arrays) to concatenate.
        :param height_concat: If True, concatenate vertically; if False, concatenate horizontally.
        :return: Concatenated image as a numpy array.
        """
        if not images:
            return np.array([])

        if height_concat:
            return cv2.vconcat(images)
        else:
            return cv2.hconcat(images)

    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    # ==================================

    base64_image = encode_image(image_path)

    response = await client.chat.completions.create(
        model="default",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prefix},
                    # {"type": "image_url", "image_url": {"url": image_path}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    {"type": "text", "text": suffix},
                ],
            }
        ],
        temperature=0,
        max_completion_tokens=sampling_params["max_new_tokens"],
        max_tokens=sampling_params["max_new_tokens"],
    )
    return sample, response.choices[0].message.content


async def process_sample_with_semaphore(
    semaphore: asyncio.Semaphore, client: Any, sample: dict, sampling_params: dict
) -> Tuple[dict, str]:
    """Wrap process_sample with a semaphore for concurrency control."""
    async with semaphore:
        return await process_sample(client, sample, sampling_params)


async def eval_mmmu(args) -> None:
    """Main evaluation loop with concurrency control."""
    eval_args = EvalArgs.from_cli_args(args)
    sampling_params = get_sampling_params(eval_args)
    samples = prepare_samples(eval_args)
    answer_dict = {}
    out_samples = {}
    client = openai.AsyncOpenAI(
        api_key="sk", base_url=f"http://127.0.0.1:{args.port}/v1"
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    start = time.perf_counter()
    base_url = f"http://127.0.0.1:{args.port}"

    if args.profile:
        print("Starting profiler...")
        profile_output = await async_request_profile(
            api_url=f"{base_url}/start_profile"
        )
        if profile_output.success:
            print("Profiler started")

        samples = samples[: args.profile_number]

    tasks = [
        process_sample_with_semaphore(semaphore, client, sample, sampling_params)
        for sample in samples
    ]

    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        sample, response = await coro
        process_result(response, sample, answer_dict, out_samples)

    if args.profile:
        print("Stopping profiler...")
        profile_output = await async_request_profile(api_url=f"{base_url}/stop_profile")
        if profile_output.success:
            print("Profiler stopped")

    print(f"Benchmark time: {time.perf_counter() - start}")
    print(f"Requests per second: {len(samples) / (time.perf_counter() - start)} req/s")
    args.output_path = f"./val_sglang.json"
    save_json(args.output_path, out_samples)
    eval_result(model_answer_path=args.output_path, answer_dict=answer_dict)


def parse_args():
    parser = argparse.ArgumentParser()
    EvalArgs.add_cli_args(parser)
    args = add_common_sglang_args_and_parse(parser)
    return args


def main():
    args = parse_args()
    asyncio.run(eval_mmmu(args))


if __name__ == "__main__":
    main()
