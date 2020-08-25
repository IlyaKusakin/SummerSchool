import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import pickle

import torch
from torch import nn
import torch.nn.functional as F


import re
import nltk

from nltk.stem.porter import *
stemming = PorterStemmer()

from nltk.tokenize import WordPunctTokenizer
tokenizer = WordPunctTokenizer()

nltk.download('stopwords')
stopWords =nltk.corpus.stopwords.words()
stopWords.append("the")
stopWords.append('i')
stopWords.append("a")
stopWords.append('visitor')
stopWords.append('chat')
stopWords.append('transcript')



from flask import Flask
from flask import request
import requests
from flask import jsonify

import os
import json
from ast import literal_eval
import traceback

application = Flask(__name__)


model = nn.Sequential(
    nn.Linear(1492,800),
    nn.ReLU(),
    nn.Dropout(0.8),

    nn.BatchNorm1d(800),
    nn.Linear(800,500),
    nn.ReLU(),
    nn.Dropout(0.8),

    nn.BatchNorm1d(500),
    nn.Linear(500,200),
    nn.ReLU(),
    nn.Dropout(0.8),
    
    nn.BatchNorm1d(200),
    nn.Linear(200,3))


#загружаем модели из файла
model.load_state_dict(torch.load("./models/weights_86acc.pt"))
vec = pickle.load(open("./mytfidf.pickle", "rb"))


# тестовый вывод
@application.route("/")  
def hello():
    resp = {'message':"Hello World!"}
    
    response = jsonify(resp)
    
    return response

# предикт категории
#{"user_message":"example123rfssg gsfgfd"}
@application.route("/categoryPrediction" , methods=['GET', 'POST'])  
def registration():
    resp = {'message':'ok'
           ,'category': -1
           }

    try:
        getData = request.get_data()
        json_params = json.loads(getData) 
        
        message = json_params['user_message']
        message_ = text_preprocessing(message)
        
        vector =vec.transform([message_]).toarray()
        vector = torch.FloatTensor(vector)
        
        #напишите прогноз и верните его в ответе в параметре 'prediction'
        model.eval()
        prediction = F.softmax(model(vector)).detach()[0].tolist()
        
        resp['category'] = prediction

        
    except Exception as e: 
        print(e)
        resp['message'] = e
      
    response = jsonify(resp)
    
    return response


def text_preprocessing(message):
  message = re.sub('\[.*\]', '', message)
  message = re.sub("\!", '', message)
  message = re.sub("\'", '', message)
  message = re.sub('[-\’,·”–●•№~✅“=#—«"‚»|.?!:;()*^&%+/]', ' ' , message)
  message = message.replace("\s+", ' ')
  message = message.lower()

  message = tokenizer.tokenize(message)
  message = [stemming.stem(word) for word in message if word not in stopWords]

  clear_message = ''
  for word in message:
    clear_message+=' '+word

  return clear_message

        

if __name__ == "__main__":
    port = int(os.getenv('PORT', 5000))
    application.run(debug=False, port=port, host='0.0.0.0' , threaded=True)



