import json
import numpy as np
import re
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer

def load_model_tokenizer(path = "", CUDA = False):
    if CUDA == False:
        if path == "":
            model = AutoModel.from_pretrained('./model').to('cuda')
            tokenizer = AutoTokenizer.from_pretrained('uitnlp/CafeBERT')
        else:
            model = AutoModel.from_pretrained(path).to('cuda')
            tokenizer = AutoTokenizer.from_pretrained(path)
    else:
        if path == "":
            model = AutoModel.from_pretrained('./model')
            tokenizer = AutoTokenizer.from_pretrained('uitnlp/CafeBERT')
        else:
            model = AutoModel.from_pretrained(path)
            tokenizer = AutoTokenizer.from_pretrained(path)

    return model, tokenizer

def vectorize_sentence(sen, model, tokenizer):
    encoding = tokenizer(sen, return_tensors='pt').to('cuda')
    with torch.no_grad():
        output = model(**encoding)
        word_vectors = output.last_hidden_state[0, 0, :].cpu().detach().numpy()
        return word_vectors
    
def trace_context(sen_ID, data):
    context = ""
    if(sen_ID[3]=="C"):
        small_data = data[sen_ID[0]][sen_ID[1:3]][sen_ID[3:6]]
        for lesson in small_data.keys():
            if lesson != "name":
                context +="."
                context += small_data[lesson]['context']
    else:
        small_data = data[sen_ID[0]][sen_ID[1:3]]
        for chapter in small_data.keys():
            for lesson in small_data[chapter].keys():
                if lesson == sen_ID[3:6]:
                    context = small_data[chapter][lesson]['context']
                    print(context, chapter, lesson)
                    break
            break
    return context

def format_QA(QA):
    for i in range(4):
        QA['answer_options'][i] = QA['answer_options'][i][3:]
    QA['correct_answer'] = ord(QA['correct_answer'][0]) - ord('A')
    return QA

def vectorize_context_QA(context, QA, model, tokenizer):
    delimiters = "[.,;!]"  
    sens = re.split(delimiters, context)
    vec_context = []
    for sen in sens:
        new_vec_sen = vectorize_sentence(sen, model, tokenizer)
        vec_context.append(new_vec_sen)
    vec_question = vectorize_sentence(QA['question'], model, tokenizer)
    vec_answer_options = []
    for ans in QA['answer_options']:
        new_vec_ans_opt = vectorize_sentence(ans, model, tokenizer)
        vec_answer_options.append(new_vec_ans_opt)
    return sens, vec_context, vec_question, vec_answer_options

def select_sens(sens, vec_context, vec_question, e):
    i = 0
    while(True):
        if(len(sens) == 5):
            break
        if(i == len(sens)):
            break
        cosine_sim = cosine_similarity([vec_question], [vec_context[i]])
        if cosine_sim <= e:
            sens.pop(i)
            vec_context.pop(i)
        else:
            i+=1
    return sens, vec_context

def knn_labels(points, centers):
    points = np.array(points)
    centers = np.array(centers)

    similarity = cosine_similarity(points, centers)
    distance = 1 - similarity
    # Find the index of the center with the highest similarity (smallest distance)
    labels = np.argmin(distance, axis=1)


    # Compute the mean distance (cost) for each center
    min_distances = distance[np.arange(distance.shape[0]), labels]
    costs = []
    for center_idx in range(centers.shape[0]):
        center_distances = min_distances[labels == center_idx]
        if len(center_distances) > 0:
            mean_distance = np.mean(center_distances)
        else:
            mean_distance = 0  # No points assigned to this center
        costs.append(mean_distance)
    
    return labels, costs

def predict(labels, costs, sens):
    costs = np.array(costs)
    sum_cost = costs.sum()
    percentage_answers = [
        costs[0]/sum_cost,
        costs[1]/sum_cost,
        costs[2]/sum_cost,
        costs[3]/sum_cost,
    ]
    pred_ans_ratio = max(percentage_answers)
    pred_ans = percentage_answers.index(pred_ans_ratio)
    explain = [sens[i] for i in np.where(labels == pred_ans)[0]]
            
    return pred_ans, pred_ans_ratio, explain

def export_answer(pred_ans, pred_ans_ratio, QA_key, explain, output):
    result = {
        "predicted_answer": pred_ans,
        "predicted_answer_ratio": pred_ans_ratio.tolist(),
        "QA_key": QA_key,
        "explain": explain
    }
    with open(output, 'a') as f:
        json.dump(result, f)
        f.write('\n')