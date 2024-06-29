from tqdm import tqdm
from _code import load_model_tokenizer, vectorize_context_QA, trace_context, format_QA, select_sens, knn_labels, predict, export_answer
import torch
import json
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--path", type=str, help="Path to model and tokenizer on huggingface", default="")
parser.add_argument("--output", type=str, help="Path to output file", default="output.json")
parser.add_argument("--QAs", type=str, help="QA set", default="data_full.json")
parser.add_argument("--dataset", type=str, help="Dataset SGK", default="sgk_final_new.json")
parser.add_argument("--e", type=float, help="threshold for cosine similarity in select_sens", default=0.5)
parser.add_argument("--cuda_device", type=int, help="CUDA device index", default=0)

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
print(f"Start predicting {len(QAs)} question")

# n_QA: formatted QA
# sens: list of sentences in context
# vec_x: vectorized "x"
for QA_key in tqdm(QAs, desc="Processing QAs"):
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
    export_answer(pred_ans, pred_ans_ratio, QA_key, explain, args.output)
