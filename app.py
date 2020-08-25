from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import pickle

import re
import nltk

from nltk.stem.porter import *
stemming = PorterStemmer()
    
from nltk.tokenize import WordPunctTokenizer
tokenizer = WordPunctTokenizer()


nltk.download('stopwords')
stopWords =nltk.corpus.stopwords.words()
#adding custom stopwords based on experience
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

# franework for neural nets creating
import torch
from torch import nn
import torch.nn.functional as F

# Dense neural net for category prediction
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
    
# uploading models
model.load_state_dict(torch.load("./models/weights_86acc.pt"))
vec = pickle.load(open("./models/mytfidf.pickle", "rb"))

    

def text_preprocessing(message):
    """
    Function for symbols deleting, lowcasing, tokens stemming
    
    Parameters:
        message: str
        Stock text message with rubbish
        
    Returns:
        cleaned_message: str
        Message cleaned from symbols and with stemmed words
        
    Example:
        
        Input:
            "Chat transcript: Visitor: I am attempting to verify my card 
            however the payment is not posting. All charges typically post 
            immediately with my bank.Sofia: Hello!Sofia: Please stand by online, 
            as it may take some time to resolve the issue. Ill provide you 
            with an update in a few minutes. Thank you for your patience.
            [Visitor page reloaded. New URL: https://secure.xsolla.com/paystati
             on3/?access_token=vlp0QhPAE54cVIsykIWwTE0BZp70mAQy ][Visitor page 
            reloaded. New URL: https://secure.xsolla.com/paystation3/?access_
            token=vlp0QhPAE54cVIsykIWwTE0BZp70mAQy ][Visitor page reloaded. 
            New URL: https://secure.xsolla.com/paystation3/?access_token=
            vlp0QhPAE54cVIsykIWwTE0BZp70mAQy ]"
                                            
        Output:
             "attempt verifi card howev payment post charg typic post 
             immedi bank sofia hellosofia pleas stand onlin may time 
             resolv issu provid updat minut thank patienc"                                       
   """
      
    message = re.sub('\[.*\]', '', message)
    message = re.sub("\!", '', message)
    message = re.sub("\'", '', message)
    message = re.sub('[-\’,·”–●•№~✅“=#—«"‚»|.?!:;()*^&%+/]', ' ' , message)
    message = message.replace("\s+", ' ')
    message = message.lower()
    
    message = tokenizer.tokenize(message)
    message = [stemming.stem(word) for word in message if word not in stopWords]
    
    cleaned_message = ''
    for word in message:
        cleaned_message+=' '+word
    
    return cleaned_message


application = Flask(__name__)


# test output
@application.route("/")  
def hello():
    resp = {'message':"Hello World!"}
    response = jsonify(resp)
    
    return response


@application.route("/categoryPrediction" , methods=['GET', 'POST'])  
def registration():
    """
    Function for request catching and category predicting.
    
    Returns:
        repsonse: json 
            Json with message 'ok' and array with categories probabilities
            or text of exception if erros occured.
        
    Example:
        Input:
            {"user_message":"example123rfssg gsfgfd"}
        Output:
             {"category": [ 0.28119155764579773,
                           0.500522255897522,
                           0.21828614175319672],
             "message": "ok"}                                    
    """
    
    resp = {'message':'ok'
           ,'category': -1
           }

    try:
        getData = request.get_data()
        json_params = json.loads(getData) 
        message = json_params['user_message']
        
        #message cleaning
        message_ = text_preprocessing(message)
        
        if message.replace(' ', '')=='':
            resp['category'] = [1.0, 0.0, 0.0]
        
        else:
            #message vectorizing
            vector =vec.transform([message_]).toarray()
            vector = torch.FloatTensor(vector)
                
            #inference mode
            model.eval()
            prediction = F.softmax(model(vector)).detach()[0].tolist()
                
            resp['category'] = prediction
     
    except Exception as e: 
        print(e)
        resp['message'] = e
      
    response = jsonify(resp)
    
    return response

        
if __name__ == "__main__":
    port = int(os.getenv('PORT', 5000))
    application.run(debug=False, port=port, host='0.0.0.0' , threaded=True)



