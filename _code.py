import json
import numpy as np
import re
import torch
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer

def load_model_tokenizer(path = "", device='cuda:0'):
    if path == "":
        model = AutoModel.from_pretrained('./model').to(device)
        tokenizer = AutoTokenizer.from_pretrained('uitnlp/CafeBERT')
    else:
        model = AutoModel.from_pretrained(path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(path)
    return model, tokenizer

def vectorize_sentence(sen, model, tokenizer, device='cuda:0'):
    encoding = tokenizer(sen, return_tensors='pt').to(device)
    with torch.no_grad():
        output = model(**encoding)
        word_vectors = output.last_hidden_state[0, 0, :].cpu().detach().numpy()
        return word_vectors

def trace_context(sen_ID, data):
    context = ""
    if sen_ID[3] == "C":
        small_data = data[sen_ID[0]][sen_ID[1:3]][sen_ID[3:6]]
        for lesson in small_data.keys():
            if lesson != "name":
                context += "."
                context += small_data[lesson]['context']
    else:
        found = False
        small_data = data[sen_ID[0]][sen_ID[1:3]]
        for chapter in small_data.keys():
            for lesson in small_data[chapter].keys():
                if lesson == sen_ID[3:6]:
                    context = small_data[chapter][lesson]['context']
                    found = True
                    break 
            if found:
                break
    return context

def format_QA(QA):
    for i in range(4):
        QA['answer_options'][i] = QA['answer_options'][i][3:].lower()
    QA['correct_answer'] = ord(QA['correct_answer'][0]) - ord('A')
    return QA

def vectorize_context_QA(context, QA, model, tokenizer, device='cuda:0'):
    delimiters = "[.,;!]"
    sens = re.split(delimiters, context)
    sens = [sen.strip() for sen in sens if sen != '.' or len(sen) > 10]
    vec_context = []
    valid_sens = []
    for sen in sens:
        try:
            new_vec_sen = vectorize_sentence(sen, model, tokenizer, device)
            vec_context.append(new_vec_sen)
            valid_sens.append(sen)
        except Exception as e:
            print(f"Could not vectorize sentence: '{sen}'. Error: {e}")
    vec_question = vectorize_sentence(QA['question'], model, tokenizer, device)
    vec_answer_options = [vectorize_sentence(opt, model, tokenizer, device) for opt in QA['answer_options']] 
    return valid_sens, np.array(vec_context), vec_question, np.array(vec_answer_options)

def select_sens(sens, vec_context, vec_question, threshold, device='cuda:0'):
    vec_question = torch.tensor(vec_question).to(device)
    selected_sens = []
    selected_vec_context = []
    for sen, vec in zip(sens, vec_context):
        vec = torch.tensor(vec).to(device)
        cos_sim = cosine_similarity(vec_question.cpu().numpy().reshape(1, -1), vec.cpu().numpy().reshape(1, -1))
        if cos_sim >= threshold:
            selected_sens.append(sen)
            selected_vec_context.append(vec.cpu().numpy())
    return selected_sens, np.array(selected_vec_context)

def knn_labels(vec_context, vec_answer_options, device='cuda:0'):
    vec_context = torch.tensor(vec_context).to(device)
    vec_answer_options = torch.tensor(vec_answer_options).to(device)
    labels = []
    costs = []
    for vec in vec_context:
        cos_sims = cosine_similarity(vec.cpu().numpy().reshape(1, -1), vec_answer_options.cpu().numpy())
        label = np.argmax(cos_sims)
        cost = np.max(cos_sims)
        labels.append(label)
        costs.append(cost)
    return labels, costs

def predict(labels, costs, sens):
    costs = np.array(costs)
    unique_labels = [0,1,2,3]
    sums = np.zeros(4)
    counts = np.zeros(4)
    for label, cost in zip(labels, costs):
        idx = np.where(unique_labels == label)[0][0]
        sums[idx] += cost
        counts[idx] += 1
    avg_costs = np.divide(sums, counts, out=np.zeros_like(sums), where=counts != 0)
    if avg_costs.sum() == 0:
        return 4, 0, 'null'
    sum_cost = avg_costs[0] + avg_costs[1] + avg_costs[2] + avg_costs[3]
    percentage_answers = [
        avg_costs[0]/sum_cost,
        avg_costs[1]/sum_cost,
        avg_costs[2]/sum_cost,
        avg_costs[3]/sum_cost,
    ]
    pred_ans_ratio = max(percentage_answers)
    pred_ans = percentage_answers.index(pred_ans_ratio)
    explain = ""
    i = 0
    for i in range(len(sens)):
        if(labels[i]==pred_ans):
            explain += (sens[i] +". ")
            
    return pred_ans, pred_ans_ratio, explain

def export_answer(pred_ans, pred_ans_ratio, QA_key, explain, output):
    with open(output,"r+",encoding='utf-8') as f:
        output_data = json.load(f)
    new = {'Predicted answer':pred_ans,
           'Ratio': float(round(pred_ans_ratio, 2)),
           'Explain': explain
           }
    output_data.update({QA_key:new})
    with open(output, "r+", encoding='utf-8') as f:
        json.dump(output_data,f,ensure_ascii=False,indent=4)

def process_batch(batch_keys, QAs, dataset, model, tokenizer, device, e, output):
    results = {}
    for QA_key in tqdm(batch_keys, desc=f"Processing QAs in Batch"):
        QA = QAs[QA_key]
        formated_QA = format_QA(QA)

        # Get context for QA
        context = trace_context(QA_key, dataset)

        # Vectorize context and QA
        sens, vec_context, vec_question, vec_answer_options = vectorize_context_QA(context, formated_QA, model, tokenizer, device)

        # Remove context based on cosine similarity with threshold e
        sens, vec_context = select_sens(sens, vec_context, vec_question, e, device)

        # Clustering based on cosine similarity
        labels, costs = knn_labels(vec_context, vec_answer_options, device)

        # Predict and get explanation
        pred_ans, pred_ans_ratio, explain = predict(labels, costs, sens)

        # Save predicted answer
        export_answer(pred_ans, pred_ans_ratio, QA_key, explain, output)
    return results