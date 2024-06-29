from tqdm import tqdm
from _code import load_model_tokenizer, process_batch
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import argparse

parser = argparse.ArgumentParser(description="Process QA data in batches")
parser.add_argument("--path", type=str, help="Path to model and tokenizer", default="")
parser.add_argument("--output", type=str, help="Path to output file", default="output.json")
parser.add_argument("--QAs", type=str, help="QA set", default="data_full.json")
parser.add_argument("--dataset", type=str, help="Dataset SGK", default="sgk_final_new.json")
parser.add_argument("--e", type=float, help="Threshold for cosine similarity in select_sens", default=0.5)
parser.add_argument("--cuda_device", type=int, help="CUDA device index", default=0)
parser.add_argument("--batch_size", type=int, help="Number of QAs per batch", default=100)
parser.add_argument("--num_workers", type=int, help="Number of parallel workers", default=4)

args = parser.parse_args()

# Set the device
device = torch.device(f'cuda:{args.cuda_device}' if torch.cuda.is_available() else 'cpu')

# Load model, tokenizer, QAs set and dataset
print("Load model, tokenizer")
model, tokenizer = load_model_tokenizer(args.path, device)
print("Loaded")
print("Load dataset, QAs")
with open(args.QAs, 'r+', encoding='utf-8') as json_file:
    QAs = json.load(json_file)
with open(args.dataset, 'r+', encoding='utf-8') as json_file:
    dataset = json.load(json_file)
print("Loaded")

QA_keys = list(QAs.keys())
num_batches = (len(QA_keys) + args.batch_size - 1) // args.batch_size

print(f"Start predicting {len(QAs)} question in {num_batches} batches")

# n_QA: formatted QA
# sens: list of sentences in context
# vec_x: vectorized "x"
with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
    future_to_batch = {}
    for batch_idx in range(num_batches):
        batch_start = batch_idx * args.batch_size
        batch_end = min((batch_idx + 1) * args.batch_size, len(QA_keys))
        batch_keys = QA_keys[batch_start:batch_end]
        future = executor.submit(process_batch, batch_keys, QAs, dataset, model, tokenizer, device, args.e, args.output)
        future_to_batch[future] = batch_idx

    for future in as_completed(future_to_batch):
        batch_idx = future_to_batch[future]
        batch_results = future.result()
        # Save intermediate results after each batch
        print(f"Batch {batch_idx + 1}/{num_batches} complete and saved.")

print("Processing complete. Results saved to", args.output)