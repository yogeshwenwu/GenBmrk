import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import torch.amp as amp

# Input
sentences = [
    "The weather is bad today it might rain anytime",
    "Artificial intelligence is transforming the way",
    "The movie I watched yesterday had an unexpected twist at the end",
    "You recommended a good book to read over the weekend, that was",
    "The capital of France is Paris, known for its art, culture",
    "She ordered a latte at the café and worked on her presentation",
    "The key differences between machine learning and deep learning is",
    "The traffic on my way to work this morning was",
    "Python is a versatile programming language often used in",
    "He went to the gym every day, determined to improve",
    "Quantum computing has the potential to revolutionize data encryption",
    "The latest advancements in robotics have enabled autonomous medical procedures",
    "NASA's new telescope can capture images of distant galaxies with unprecedented clarity",
    "The discovery of a new exoplanet raises questions about extraterrestrial life",
    "Meditation has been shown to improve mental clarity and reduce stress",
    "Scientists recently found a way to generate renewable energy from ocean waves",
    "The stock market saw a major shift after the latest tech industry boom",
    "Investing in cryptocurrency can be both rewarding and risky",
    "Cultural diversity enriches society by introducing new perspectives and traditions",
    "The human brain remains one of the most complex and least understood organs",
    "The ethical implications of genetic cloning continue to be widely debated",
    "If time travel were possible, would we change the past or the future",
    "The Renaissance period was a time of great artistic and intellectual growth",
    "She had a dream about a hidden city beneath the ocean waves",
    "The detective knew the case wasn’t as simple as it seemed",
    "He found an ancient map hidden inside his grandfather’s journal",
    "A simple act of kindness can brighten someone’s entire day",
    "Financial literacy is an essential skill that should be taught in schools",
    "The invention of the printing press revolutionized the spread of knowledge",
    "The old man stared at the letter, unsure if he should open it",
    "The moon shone brighter than ever before, casting an eerie glow over the city",
    "A mysterious book appeared on her doorstep with no sender address",
    "Regular exercise not only improves physical health but also boosts mood",
    "A secret tunnel beneath the library led to a forgotten underground world",
    "The radio suddenly started playing a song from the future",
    "The cat stared at the empty space, as if it could see something invisible",
    "A group of scientists accidentally opened a portal to another dimension",
    "The future of self-driving cars depends on the reliability of AI decision-making",
    "The traffic on my way to work this morning was unbearable",
    "The Eiffel Tower is one of the most famous landmarks in the world",
    "The clock struck midnight, and the entire town vanished",
    "He struggled to find the right words to express his gratitude",
    "Natural language processing allows chatbots to understand human emotions better",
    "The aroma of freshly brewed coffee filled the cozy café",
    "Sleep deprivation can negatively impact cognitive function and productivity",
    "The Great Wall of China was originally built to protect against invasions",
    "Web development and data science are two of the most popular tech fields today",
    "Artificial intelligence is expected to revolutionize many industries in the coming years",
    "The enchanted forest was said to grant wishes to those who entered with pure intentions",
    "The human body has an incredible ability to heal itself under the right conditions"
]

# Configuration
batch_size = 16
total_sentences = len(sentences)

# 1. Setup: Load Model and Tokenizer
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
device = torch.device("cpu")
model.to(device)
model.eval()

# 2. Warmup Runs 
print("Running warmup...")
warmup_batch = sentences[:batch_size] 
for _ in range(3):
    inputs = tokenizer(warmup_batch, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=50)


dataloader = DataLoader(sentences, batch_size=batch_size, shuffle=True)

# 3. Batch Inference and Benchmarking
print("Starting benchmark with bfloat16 quantization...")
total_input_tokens = 0
total_output_tokens = 0
total_time = 0
batch_latencies = []
b1 = sentences[:batch_size] 

for _ in range(3):
    ttft_start = time.time()
    inputs = tokenizer(b1, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=1)
    ttft = time.time() - ttft_start

for batch in dataloader:
    batch_start_time = time.time()
    # Tokenize the batch inside the loop
    inputs = tokenizer(batch, return_tensors="pt", padding=True).to(device)
    current_batch_size = inputs["input_ids"].shape[0]
    input_token_count = current_batch_size * inputs["input_ids"].shape[1]
    total_input_tokens += input_token_count

    # Inference with timing
    with torch.no_grad():
        with amp.autocast('cpu', dtype=torch.bfloat16):
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                return_dict_in_generate=True,
                output_scores=True
            )
    batch_time = time.time() - batch_start_time
    total_time += batch_time
    batch_latencies.append(batch_time)

    # Calculate output tokens
    output_token_count = outputs.sequences.shape[1] - inputs["input_ids"].shape[1]
    total_output_tokens += output_token_count * current_batch_size

# 5. Metrics Calculation
num_batches = len(batch_latencies)
avg_latency_per_batch = total_time / num_batches
avg_latency_per_request = total_time / total_sentences
throughput_tps = total_output_tokens / total_time  
throughput_rps = total_sentences / total_time  
ttft_estimate = total_time / (total_output_tokens + num_batches)
tpot = total_time / total_output_tokens  # Time per output token

# 6. Display Results
print("\nBenchmark Results:")
print(f"Total Sentences: {total_sentences}")
print(f"Batch Size: {batch_size}")
print(f"Number of Batches: {num_batches}")
print(f"Total Input Tokens: {total_input_tokens}")
print(f"Total Output Tokens: {total_output_tokens}")
print(f"Total Time: {total_time:.4f} seconds")
print(f"Avg Latency per Batch: {avg_latency_per_batch:.4f} seconds")
print(f"Avg Latency per Request: {avg_latency_per_request:.4f} seconds")
print(f"Throughput (TPS): {throughput_tps:.2f} tokens/second")
print(f"Throughput (RPS): {throughput_rps:.2f} requests/second")
print(f"Time to First Token (TTFT, estimated): {ttft_estimate:.6f} seconds")
print(f"Time to First Token (TTFT): {ttft:.6f} seconds")
print(f"Time per Output Token (TPOT): {tpot:.6f} seconds")
print("\nBatch Latencies:")
for i, latency in enumerate(batch_latencies):
    print(f"Batch {i+1}: {latency:.4f} seconds")