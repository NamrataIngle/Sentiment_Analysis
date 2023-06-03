#!/usr/bin/env python
# coding: utf-8

import pickle

#libraries
import pandas as pd # data processing
import numpy as np # linear algebra

#feature engineering
from sklearn import preprocessing

# data transformation
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


import warnings
warnings.filterwarnings('ignore')


#from sklearn.svm import SVC

import pickle as pickle5
import streamlit as st
st.title("Twiter Symentic Analysis")
message = st.text_area("Please Enter your text")
def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://ts2.space/wp-content/uploads/2023/04/mfrack_Revolutionizing_Healthcare_with_AI_8a48a065-943a-4913-8617-e0e840c57612-1024x574.jpeg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url()

# Loading data
data=pd.read_csv('tweet.csv')

df['class'].unique()

st.title("Predict for sentiment")


'''from sklearn.preprocessing import LabelEncoder
lr = LabelEncoder()
df['class']= lr.fit_transform(df['class'])
label_list = df['label'].tolist()

# Dividing data into Features(X) & Target(y)
from gensim.models import Word2Vec

# Tokenize the text
tokenized_text = [text.split() for text in text_list]

# Train the Word2Vec model
model = Word2Vec(sentences=tokenized_text,vector_size=500,window=5,min_count=1,workers=4)

# Function to generate the document vector by averaging word vectors
def document_vector(model, doc):
    doc_vector = np.zeros(model.vector_size)
    count = 0
    for word in doc:
        if word in model.wv:
            doc_vector += model.wv[word]
            count += 1
    if count != 0:
        doc_vector /= count
    return doc_vector

# Generate document vector for each tweet
text_vectors= [document_vector(model,text)for text in tokenized_text]

# Convert text_vector to numpy array
X = np.array(text_vectors)

# Assuming 'labels' is a list containing the corresponding labels for each tweet
y = np.array(label_list)

# Train-Test Split 
#Train test split will be a 70:30 ratio respectively.
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

from sklearn.model_selection import GridSearchCV

## Creating the instance of Logistic Regression model
log_reg = LogisticRegression(C=30.0,solver='newton-cg',multi_class='multinomial',random_state=42)

## Train the model 
log_reg.fit(X_train_lr,y_train_lr)

## predict the label for test set
y_pred_lr = log_reg.predict(X_test_lr)'''


'''#SVM Clasification
svm = SVC(C=1, kernel='linear')         #bydefault kernel=rbf      C=to control soft margin
svm1=svm.fit(X_train,y_train)
result_svm = svm1.score(X_test,y_test)'''


'''filename = 'df.pkl'
pickle.dump(svm, open(filename,'wb'))
pickled_model=pickle.load(open('df.pkl','rb'))
pickled_model.fit(X_train,y_train)
pk=pickled_model.predict(X_test'''


import streamlit as st
if st.button('Predict'):
    df={'cap_shape':m_cap_shape,'cap_surface':m_cap_surface,' cap_color':m_cap_color,' bruises':m_bruises,' odor':m_odor,'gill_spacing':m_gill_spacing,'gill_size':m_gill_size,' gill_color':m_gill_color,' stalk_shape':m_stalk_shape,' stalk_root':m_stalk_root,'stalk_surface_above_ring':m_stalk_surface_above_ring,'stalk_color_above_ring':m_stalk_color_above_ring,' ring_type':m_ring_type,' spore_print_color':m_spore_print_color,' population':m_population,'habitat':m_habitat}
  

   
    df1=pd.DataFrame(df,index=[1])
    df1==pd.get_dummies(df1)
    predictions=pickled_model.predict(df1)
    
    if predictions.any()==0:
        prediction_value = 'Figurative'
    elif prediction.any()==1:
        prediction_value = 'Irony'
    elif prdiction.any()==2:
        prediction_value = 'Regular'
    else:
        prediction_value = 'Sarcastic'
        
    
    st.title(prediction_value)
        
