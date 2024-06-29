import json
import argparse
from tqdm import tqdm
import torch
from _code import load_model_tokenizer, format_QA, trace_context, vectorize_context_QA, select_sens, knn_labels, predict, export_answer

def save_intermediate_results(output_path, results):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

parser = argparse.ArgumentParser(description="Process QA data in batches")
parser.add_argument("--path", type=str, help="Path to model and tokenizer", default="")
parser.add_argument("--output", type=str, help="Path to output file", default="output.json")
parser.add_argument("--QAs", type=str, help="QA set", default="data_full.json")
parser.add_argument("--dataset", type=str, help="Dataset SGK", default="sgk_final_new.json")
parser.add_argument("--e", type=float, help="Threshold for cosine similarity in select_sens", default=0.7)
parser.add_argument("--cuda_device", type=int, help="CUDA device index", default=0)
parser.add_argument("--batch_size", type=int, help="Number of QAs per batch", default=17000)

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
results = {}

for batch_idx in tqdm(range(num_batches), desc="Processing Batches"):
    batch_start = batch_idx * args.batch_size
    batch_end = min((batch_idx + 1) * args.batch_size, len(QA_keys))
    batch_keys = QA_keys[batch_start:batch_end]

    for QA_key in tqdm(batch_keys, desc=f"Processing QAs in Batch {batch_idx + 1}/{num_batches}"):
        QA = QAs[QA_key]
        n_QA = format_QA(QA)

        # Get context for QA
        context = trace_context(QA_key, dataset)

        # Vectorize context and QA
        sens, vec_context, vec_question, vec_answer_options = vectorize_context_QA(context, QA, model, tokenizer, device)

        # Remove context based on cosine similarity with threshold args.e
        sens, vec_context = select_sens(sens, vec_context, vec_question, args.e, device)

        # Clustering based on cosine similarity
        # costs = average of all cosine sim from a points to it true label
        labels, costs = knn_labels(vec_context, vec_answer_options, device)

        # Predict and get explanation
        pred_ans, pred_ans_ratio, explain = predict(labels, costs, sens)

        # Save predicted answer
        results[QA_key] = {
            'predicted_answer': pred_ans,
            'predicted_answer_ratio': pred_ans_ratio,
            'explanation': explain
        }

    # Save intermediate results after each batch
    save_intermediate_results(args.output, results)

print("Processing complete. Results saved to", args.output)
