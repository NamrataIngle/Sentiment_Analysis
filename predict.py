# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import streamlit as st
import pickle
import textblob
import pandas as pd
#from cleantext import clean
#from tensorflow.keras.preprocessing.text import Tokenizer
import time
import re
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
nltk.download('stopwords')
#from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer

file = 'clf.pkl'
#model = pickle.load(file)

with open(file, 'rb') as file:
    model = pickle.load(file)

text = ''
text1 = ''

st.header('Sentiment Analysis \U0001F606')
with st.title('Analyze Text'):
	text = st.text_input('Text here: ')
    
if text:
	text1=text
	blob = TextBlob(text)
    
def Analyse(text):
    prediction = model.predict('class')
    return prediction

def display_sarcastic_remark(remark):
    st.title(remark)
    time.sleep(0.1)

    


#plt.figure(figsize = (20,20))
    
if(text1!=""):
    st.title("Cleaned Text")
    text1 = re.sub('((www.[^s]+)|(https?://[^s]+))|(http?://[^s]+)', '',text1)
    tknzr = TweetTokenizer(strip_handles=True)
    text1=tknzr.tokenize(text1)
    text1=str(text1)
    text1=re.sub(r'[^a-zA-Z0-9\s]', '', text1)
    #text1=clean(text1, clean_all= False, extra_spaces=True ,stopwords=True ,lowercase=True ,numbers=True , punct=True)
    st.write(text1)
    
with open(file, 'rb') as file:
    model = pickle.load(file)
unseen_tweets=[text]
unseen_df=pd.DataFrame(unseen_tweets)
unseen_df.columns=["Unseen"]

vectorizer = TfidfVectorizer(max_features=200)

stopwords_set = set(stopwords.words('english'))

X_test = vectorizer.transform(unseen_tweets)
y_pred = model.predict(X_test)


if text!="":
    if(y_pred==0):
        remark = "That's Figurative!😄"
        display_sarcastic_remark(remark)
    if(y_pred==1):
        remark = "That's Irony!😏"
        display_sarcastic_remark(remark)
    if(y_pred==2):
        remark = "That's Regular!😐"
        display_sarcastic_remark(remark)
    if(y_pred==3):
        remark = "That's Sarcasm!🙃"
        display_sarcastic_remark(remark)
else:
    st.write(text1)
    remark = "No Words to Analyze"
    display_sarcastic_remark(remark)

    
