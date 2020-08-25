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
        pass
        
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
  message = [stemming.stem(word) for word in message]# if word not in stopWords]

  clear_message = ''
  for word in message:
    clear_message+=' '+word

  return clear_message

        

if __name__ == "__main__":
    port = int(os.getenv('PORT', 5000))
    application.run(debug=False, port=port, host='0.0.0.0' , threaded=True)



