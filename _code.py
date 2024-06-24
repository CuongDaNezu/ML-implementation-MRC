from transformers import AutoModel, AutoTokenizer
import torch
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
import numpy as np
import re

def load_model_tokenizer():
    model = AutoModel.from_pretrained('./model')
    tokenizer = AutoTokenizer.from_pretrained('uitnlp/CafeBERT')
    return model, tokenizer

def vectorize_sentence(sen, model, tokenizer):
    encoding = tokenizer(sen, return_tensors='pt')
    with torch.no_grad():
        output = model(**encoding)
        word_vectors = output.last_hidden_state[0, 0, :].cpu().detach().numpy()
        return word_vectors
    
def trace_context(sen_ID, data):
    need_check_ID = sen_ID[3]
    if(sen_ID=="C"):
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
    QA['answer_options'][1]=QA['answer_options'][1][3:]
    QA['answer_options'][2]=QA['answer_options'][2][3:]
    QA['answer_options'][3]=QA['answer_options'][3][3:]
    QA['answer_options'][4]=QA['answer_options'][4][3:]
    if(QA['correct_answer'][0]=="A"):
        QA['correct_answer']=0
    elif(QA['correct_answer'][0]=="B"):
        QA['correct_answer']=1
    elif(QA['correct_answer'][0]=="C"):
        QA['correct_answer']=2
    else:
        QA['correct_answer']=3
    return QA

def vectorize_context(context, QA, model, tokenizer):
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

def select_sens(sens, vec_context, vec_question):
    #threads hold: cosine:
    e = 0.6
    
    i = 0
    while(True):
        if(len(sens) == 5):
            break
        if(i == len(sens)):
            break
        cosine_sim = cosine_similarity(vec_question, vec_context[i])
        if cosine_sim <= e:
            sens.pop(i)
            vec_context.pop(i)
        else:
            i+=1
    return sens, vec_context, vec_question

def knn_labels(points, centers):
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
    sum_cost = costs.sum()
    percentage_answers = [
        costs[0]/sum_cost,
        costs[1]/sum_cost,
        costs[2]/sum_cost,
        costs[3]/sum_cost,
    ]
    pred_ans_ratio = max(percentage_answers)
    pred_ans = percentage_answers.index(pred_ans_ratio)
    explain = ""
    i = 0
    while(True):
        if(i==len(sens)):
            break
        if(labels[i]==pred_ans):
            explain += (sens[i] +". ")
    return pred_ans, pred_ans_ratio, explain