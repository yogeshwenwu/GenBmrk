import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from torch.cpu.amp import autocast  # For CPU-compatible AMP

# 1. Setup: Load Model and Tokenizer
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

# Move model to CPU 
device = torch.device("cpu")
model.to(device)

# Sample batch of prompts
batch_prompts = [
    "What is the capital of France?",
    "Tell me a short story about a cat.",
    "Explain quantum physics in simple terms."
]
batch_size = len(batch_prompts)

# 2. Warmup Runs
print("Running warmup...")
# for _ in range(3):
#     inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(device)
#     with torch.no_grad():
#         _ = model.generate(**inputs, max_new_tokens=50)

ttft_start = time.time()
inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(device)

with torch.no_grad():
   outputs = model.generate(
        **inputs,
        max_new_tokens=1,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True
)

print(outputs.keys())
print(outputs.scores)
print(outputs.scores[0].shape)
ttft_time = time.time() - ttft_start

# 3. Token Counting and Benchmarking
# start_time = time.time()
# inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(device)
# input_token_count = inputs["input_ids"].shape[1] * batch_size  # Tokens per prompt * batch size
# print(f"Total input tokens: {input_token_count}")

# # 4. Quantization with bfloat16 and Batch Inference
# print("Starting benchmark with bfloat16 quantization...")

# # Variables for TTFT and TPOT
# output_token_count = 0



# with torch.no_grad():
#     # Generate outputs
#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=50,
#         return_dict_in_generate=True,
#         output_scores=True
#     )
# with torch.no_grad():
#     with autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True):  # Quantize to bfloat16
#         # Generate outputs
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=50,
#             return_dict_in_generate=True,
#             output_scores=True
#         )

# Calculate output tokens
# output_token_count = outputs.sequences.shape[1] - inputs["input_ids"].shape[1]
# total_output_tokens = output_token_count * batch_size
# print(outputs)
# print(f"Total output tokens: {total_output_tokens}")

# # Measure total time
# total_time = time.time() - start_time

# # Time per output token
# tpot = total_time / total_output_tokens  

# # 5. Metrics Calculation
# latency_per_batch = total_time / batch_size  # per batch
# throughput_tps = total_output_tokens / total_time  
# throughput_rps = batch_size / total_time  

# # 6. Display Results
# print("\nBenchmark Results:")
# print(f"Batch Size: {batch_size}")
# print(f"Total Time for Batch: {total_time:.4f} seconds")
# print(f"Latency per Batch: {latency_per_batch:.4f} seconds")
# print(f"Throughput (TPS): {throughput_tps:.2f} tokens/second")
# print(f"Throughput (RPS): {throughput_rps:.2f} requests/second")
# print(f"Time to First Token (TTFT): {ttft_time:.6f} seconds")
# print(f"Time per Output Token (TPOT): {tpot:.6f} seconds")