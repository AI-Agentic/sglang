import base64
import time
import concurrent.futures
from openai import OpenAI
from tqdm import tqdm

# Global configuration
CONCURRENT_REQUESTS = 10  # Adjust this number as needed
TOTAL_CALLS = 100        # Total number of API calls to make

client = OpenAI(base_url="http://127.0.0.1:30000/v1")

def encode_image(image_path):
    """Encode image to base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def call_api():
    """Make a single API call"""
    try:
        base64_image = encode_image("./example_image.png")
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-VL-3B-Instruct",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    }
                ]
            }],
            max_tokens=300,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

def run_concurrent_requests():
    """Execute concurrent API requests"""
    print(f"Starting {TOTAL_CALLS} requests with {CONCURRENT_REQUESTS} concurrent workers...")
    
    start_time = time.time()
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENT_REQUESTS) as executor:
        # Submit all tasks
        futures = [executor.submit(call_api) for _ in range(TOTAL_CALLS)]
        
        # Process completed tasks with progress bar
        for future in tqdm(concurrent.futures.as_completed(futures), total=TOTAL_CALLS):
            result = future.result()
            results.append(result)
    
    end_time = time.time()
    total_time = end_time - start_time
    throughput = TOTAL_CALLS / total_time
    
    print(f"\nCompleted {TOTAL_CALLS} requests in {total_time:.2f} seconds")
    print(f"Throughput: {throughput:.2f} requests/second")
    
    return results

if __name__ == "__main__":
    # Test single request first
    print("Testing single request...")
    single_result = call_api()
    print(f"Single request result: {single_result}\n")
    
    # Run concurrent requests
    all_results = run_concurrent_requests()
    
    # Optional: Print first few results
    print(f"\nFirst 3 results:")
    for i, result in enumerate(all_results[:3]):
        print(f"{i+1}: {result}")