import json
import torch
from tqdm import tqdm
from _code import load_model_tokenizer, trace_context, format_QA, vectorize_context_QA, select_sens, knn_labels, predict, export_answer

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--path", type = str, help = "Path to model and tokenizer on huggingface", default = "")
parser.add_argument("--output", type = str, help = "Path to output file", default = "output.json")
parser.add_argument("--QAs", type = str, help = "QA set", default = "data_full.json")
parser.add_argument("--dataset", type = str, help = "Dataset SGK", default = "sgk_final_new.json")
parser.add_argument("--e", type = float, help = "threadhold for cosine similarity in select_sens", default = 0.5)
parser.add_argument('--CUDA', action='store_true', help = "Whether to use CUDA or hot")
args = parser.parse_args()

#Load model, tokenizer, QAs set and dataset
print("Load model, tokenizer")
if args.overwrite:
    CUDA = True
else:
    CUDA = False
model, tokenizer = load_model_tokenizer(args.path, CUDA)
print("Loaded")
print("Load dataset, QAs")
with open(args.QAs, 'r+', encoding = 'utf-8') as json_file:
    QAs = json.load(json_file)
with open(args.dataset, 'r+', encoding = 'utf-8') as json_file:
    dataset = json.load(json_file)
print("Loaded")
print(f"Start predicting {len(QAs)} question")

#n_QA: formated QA
#sens: list of sentences in context
#vec_x: vectorized "x"
for QA_key in tqdm(QAs, desc="Processing QAs"):
    QA = QAs[QA_key]
    n_QA = format_QA(QA)

    #get context for QA
    context = trace_context(QA_key, dataset)

    #vectorize context and QA
    sens, vec_context, vec_question, vec_answer_options = vectorize_context_QA(context, QA, model, tokenizer)

    #remove context based on consine similarity with threadhold args.e
    sens, vec_context = select_sens(sens, vec_context, vec_question, args.e)

    #clustering based on cosine similarity
    #costs = average of all cosine sim from a points to it true label
    labels, costs = knn_labels(vec_context, vec_answer_options)

    #Predict and get explain
    pred_ans, pred_ans_ratio, explain = predict(labels, costs, sens)

    #save predicted answer
    export_answer(pred_ans, pred_ans_ratio, QA_key, explain, args.output)


