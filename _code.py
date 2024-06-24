from transformers import AutoModel, AutoTokenizer
import torch
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import re

def knn_labels(points, centers):
    distance = euclidean_distances(points, centers)
    distance = distance**2
    labels = np.argmin(distance, axis=1)
    return labels

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
    return vec_context, vec_question, vec_answer_options