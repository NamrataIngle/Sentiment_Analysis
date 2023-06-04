import streamlit as st
import pickle
from textblob import TextBlob
import pandas as pd
import cleantext
import numpy as np
import re
from nltk.tokenize import TweetTokenizer
import matplotlib.pyplot as plt
import nltk
nltk.download('all')
from nltk.corpus import stopwords

file = open('clf.pkl','rb')
#model = pickle.load(file)

with open(filename1, 'rb') as file:
    clf = pickle.load(file)

text = ''
text = ''
def main():
	st.title('Sentiment Analysis')
	
with st.title('Analyze Text'):
	text = st.text_input('Text here: ')

if text:
	text1=text
	blob = TextBlob(text)
	
if(text1!=""):
    st.title("Cleaned Text")
    text1 = re.sub('((www.[^s]+)|(https?://[^s]+))|(http?://[^s]+)', '',text1)
    tknzr = TweetTokenizer(strip_handles=True)
    text1=tknzr.tokenize(text1)
    text1=str(text1)
    text1=re.sub(r'[^a-zA-Z0-9\s]', '', text1)
    text1=cleantext.clean(text1, clean_all= False, extra_spaces=True ,stopwords=True ,lowercase=True ,numbers=True , punct=True)
    st.write(text1)
	
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

if __name__ == '__main__':
	main()
