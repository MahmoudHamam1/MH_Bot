
# coding: utf-8

import json
from flask import Flask,jsonify,request,render_template
import nltk
import numpy as np
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

f=open('input.txt','r',errors = 'ignore')
raw=f.read()
raw=raw.lower()

sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences 
lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]

# Checking for greetings
def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

# Generating response
def response(user_response):
    robo_response=''
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-3]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-3]
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you please try again..."
        return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx]
        return robo_response


app = Flask(__name__)

@app.route('/')
def index():
     data={"Code":"200"}
     return jsonify(data)

@app.route("/api/chat",methods=["GET"])
def func():
    word_tokens = nltk.word_tokenize(raw)# converts to list of words
    user_response = request.args.get('question')
    user_response=user_response.lower()
    if(user_response!='bye'):
        if('thanks' in user_response or 'thank you' in  user_response ):
            data={"Code":200,"Message":"You are welcome..."}
            return jsonify(data)
        else:
            if(greeting(user_response)!=None):
                data={"Code":200,"Message":greeting(user_response)}
                return jsonify(data)
            else:
                sent_tokens.append(user_response)
                word_tokens= word_tokens + nltk.word_tokenize(user_response)
                data={"Code":200,'Message':response(user_response)}
                sent_tokens.remove(user_response)
                return jsonify(data)
    else:
        data={"Code":300,"Message":"Bye! take care.."}
        return jsonify(data)   

#################################### For solving cross ##########################
@app.after_request
def add_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    return response

###################################  Runnting the server #################################################
if __name__ == '__main__':
    app.run(host="127.0.0.1",port=9090)