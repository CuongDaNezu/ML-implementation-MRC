from transformers import AutoModel, AutoTokenizer
import torch
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np

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

def select_top20(quesion, model, tokenizer):
