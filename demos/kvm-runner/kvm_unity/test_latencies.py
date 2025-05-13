from openai import OpenAI
import time

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
parser.add_argument("--num_times", type=int, default=100)
parser.add_argument("--num_tokens", type=int, default=1000)
parser.add_argument("--input_text", type=str, default="Tell me a joke about cookies!")
parser.add_argument("--server", type=str, default="kvm")

args = parser.parse_args()

URLS = {
    "vllm": "http://0.0.0.0:3333/v1",
    "sglang": "http://0.0.0.0:3334/v1",
    "kvm": "http://0.0.0.0:3335/v1",
}

client = OpenAI(
    api_key='fake-key',
    base_url=URLS[args.server],
)

def go(t=100, input_text=""):
    return client.completions.create(
        model="meta-llama/Llama-3.2-1B-Instruct",
        prompt=input_text,
        max_tokens=t,
        temperature=0,
        n=1,
        extra_body={"ignore_eos": True},
    )

num_times = args.num_times
num_tokens = args.num_tokens
input_text = args.input_text
total_time = 0
total_tokens = 0
for i in range(num_times):
    time_start = time.time()
    out = go(num_tokens, input_text)
    # print num tokens generated
    time_end = time.time()
    total_time += time_end - time_start
    total_tokens += out.usage.completion_tokens

avg_time = total_time / num_times
print(f"Total time: {total_time} seconds")
print(f"Average time: {avg_time} seconds")
print(f"Average tokens per second: {total_tokens / total_time}")

dummy_time = 0
for i in range(num_times):
    time_start = time.time()
    out = go(1)
    # print num tokens generated
    time_end = time.time()
    dummy_time += time_end - time_start

dummy_avg_time = dummy_time / num_times
print(f"Total time for dummy tokens: {dummy_time} seconds")
print(f"Average time for dummy tokens: {dummy_avg_time} seconds")

print()
print()

adjusted_total_time = total_time - dummy_time
adjusted_avg_time = adjusted_total_time / num_times
print(f"Adjusted total time: {adjusted_total_time} seconds")
print(f"Adjusted average time: {adjusted_avg_time} seconds")

adjusted_tokens_per_second = total_tokens / adjusted_total_time
print(f"Adjusted tokens per second: {adjusted_tokens_per_second}")
