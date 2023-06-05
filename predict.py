# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
#import pandas as pd
#from prediction import predict

file = open('clf.pkl','rb')
model = pickle.load(file)       

vectoriser1 = TfidfVectorizer(max_features=500)


def preprocess_tweet(tweet):
    # Remove any URLs
    tweet = re.sub(r'http\S+', '', tweet)
    # Remove any non-alphanumeric characters
    tweet = re.sub(r'[^\w\s]', '', tweet)
    # Remove any digits
    tweet = re.sub(r'\d+', '', tweet)
    # Remove any mentions
    tweet = re.sub(r'@\w+', '', tweet)
    
    # Tokenization
    tokens = nltk.word_tokenize(tweet)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.lower() not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)

def main():
    # Set the page title
    st.title("Sentiment Analysis")
    st.markdown('Toy model to play to classify text into(Figurative,Irony,Regular,Sarcasm)')
    
    # Get user input
    tweet_input = st.text_input("Text here:")
    
    if st.button("Analyse text"):
        if tweet_input:
            # Preprocess the input tweet text
            preprocessed_text = preprocess_tweet(tweet_input)
            
            # Apply TF-IDF transformation
            tfidf = vectoriser1.transform([preprocessed_text])
            
            # Make prediction using the trained classifier
            prediction = model.predict(tfidf)[0]
            
            # Display the prediction
            st.write("Prediction:", prediction)
        else:
            st.write("No text to analyse")
    
if __name__ == "__main__":
    main()
    

    
