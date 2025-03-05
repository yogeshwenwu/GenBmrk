# Import necessary packages
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

# Load the Tokenizer & Model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", padding_side='left')
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.bfloat16)
tokenizer.pad_token = tokenizer.eos_token
device = "cpu"  # device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

#model

# Input prompt as batch
sentences = [
    "The weather is bad today it might rain anytime",
    "Artificial intelligence is transforming the way ",
    "The movie I watched yesterday had an unexpected twist at the end ",
    "you recommended a good book to read over the weekend, that was",
    "The capital of France is Paris, known for its art, culture ",
    "She ordered a latte at the café and worked on her presentation ",
    "the key differences between machine learning and deep learning is ",
    "The traffic on my way to work this morning was ",
    "Python is a versatile programming language often used in ",
    "He went to the gym every day, determined to improve"
]

max_new_token = 20
batch_size = len(sentences)

# tokenization and  it's time
start_time = time.time()
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').input_ids
# print(encoded_input.shape[1])
var1 = [i.shape[0] for i in encoded_input] # input token size
# print(var1)-----

# # warmup run
# with torch.no_grad():
#     _ = model.generate(
#     **encoded_input,
#     max_new_tokens=max_new_token,
#     num_beams=4,
#     do_sample = True,
#     temperature = 0.7,
#     repetition_penalty=1.2,
#     return_dict_in_generate=True,
#     output_scores=True,
# )

# # TTFT time
ttft_start = time.time()
with torch.no_grad():
    outputs = model.generate(
    encoded_input,
    max_new_tokens=1,
    num_beams=1,
    do_sample = True,
    temperature = 0.7
)
ttft_end = time.time()

# TPOT time
start_time = time.time()
with torch.no_grad():
    outputs = model.generate(
    encoded_input,
    max_new_tokens=max_new_token,
    num_beams=1,
    do_sample = True,
    temperature = 0.7
)
var2 = [i.shape[0] for i in outputs] # output token size
# print(var2) -----
# print(outputs)

# # decoding the output and it's time 
generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True) # for batch decoding
end_time = time.time()
# print(generated_texts) ---

# # need to remove input tokens from output tokens
num_tokens_generated = (var2[0] - var1[0]) * batch_size
# doubt: this is for token generated in one sentence, should we do this for all the generated tokens?

ttft_time = (ttft_end - ttft_start)
tpot_time = ((end_time - start_time) - ttft_time) / ((num_tokens_generated / batch_size) - 1)

print(f"Number of Tokens generated:{num_tokens_generated}") # 200
print(f"Number of sentences: {batch_size}") # 10

# latency_perBatch = total_time / batch_size # per-batch = (tot_time/ num_itr)
latency = end_time - start_time # entire batch = tot_time
tps = num_tokens_generated / latency 
rps = batch_size / latency

# cross verify latency, ttft, tpot using the below formula
crossVerify_latency = ttft_time + (tpot_time * (max_new_token - 1))
# if (crossVerify_latency == latency - enc_time - decode_time):
if (crossVerify_latency == latency):
    print("Correct Latency") # if correct 
else:
    print("Incorrect Latency")
print()

# # print Preformance measures    
# # print(f"Input tokens: {input_token}")
print(f"CVLatency: {crossVerify_latency * 1000:.4f} ms")
print(f"Total Time: {latency:.4f} s")
print(f"Latency: {latency * 1000:.4f} ms")
print(f"TTFT: {ttft_time * 1000:.4f} ms")
print(f"TPOT: {tpot_time * 1000:.4f} ms")
print(f"TPS: {tps:.4f} tps")
print(f"RPS: {rps:.4f} rps")




# # To print the parameters with datatypes
# for name, param in model.named_parameters():
#      print(f"Parameter: {name}, Shape: {param.shape}, Data Type: {param.dtype}")