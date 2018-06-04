#To run : python matching_based_chatbot.py
#From the browser 123.45.67.890:8555

import pandas as pd
import numpy as np
from nltk.stem.porter import *
from nltk.corpus import stopwords
import difflib
import re
from flask import Flask
from flask import render_template,jsonify,request
from flask_cors import CORS, cross_origin
import requests
import random

app = Flask(__name__)
CORS(app, support_credentials=True)
app.secret_key = '12345'


stop = set(stopwords.words('english'))
data = pd.read_csv("data.csv")
#data should have atleaset 2 columns: 'question' and 'abswer'

stemmer = PorterStemmer()
data['cleaned_question'] = data['question'].apply(lambda x : ' '.join([stemmer.stem(tagged_word) for tagged_word in re.sub(r'([^\s\w]|_)+', '',str(x).strip().lower()).split()]) )

@app.route('/')
@cross_origin(supports_credentials=True)
def hello_world():
    return render_template('home.html')


@app.route('/chat',methods=["POST"])
@cross_origin(supports_credentials=True)
def chat():
    try:
        user_message = request.form["text"]
        if user_message.lower().strip() in ["hey","hello","hi"]:
            return jsonify({"status":"success","response":random.choice(["hey","hello","hi"])})
        if user_message.lower().strip() in ["bye","it was nice talking to you","see you","ttyl", "goodbye", "good bye", "take care"]:
            return jsonify({"status":"success","response":random.choice(["Bye","It was nice talking to you","See you","Ttyl", "GoodBye!", "Take care"])})
        if user_message.lower().strip() in ["cool","i know you would like it"]:
            return jsonify({"status":"success","response":random.choice(["Cool","I know you would like it"])})
        closest_match = difflib.get_close_matches(' '.join([stemmer.stem(tagged_word) for tagged_word in re.sub(r'([^\s\w]|_)+', '',str(user_message).strip().lower()).split()]), data['cleaned_question'])
        response_text = data[data['cleaned_question'] == closest_match[0]].iloc[0,:]['answer']
        print(response_text)
        if response_text != '':
            return jsonify({"status":"success","response":response_text})
        else:
            return jsonify({"status":"success","response": "Sorry! I couldn't get you"})
    except Exception as e:
        print(e)
        return jsonify({"status":"success","response":"Sorry I am not trained to do that yet..."})
        
    
    
app.config["DEBUG"] = True
if __name__ == "__main__":
    app.run(host="123.45.67.890_OR_localhost",debug=True, port=8555)
    
