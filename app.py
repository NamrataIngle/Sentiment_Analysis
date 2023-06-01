import pandas as pd
import streamlit as st 
from textblob import TextBlob

st.title('Model Deployment: Sentiment Analysis')
text = st.text_area("Please Enter your text")

if st.button("Analyze the Sentiment"):
  blob = TextBlob(message)
  result = blob.sentiment
  polarity = result.polarity
  subjectivity = result.subjectivity
  if polarity < 0:
    st.warning("The entered text has negative sentiments associated with it"+str(polarity))
    rain(
    emoji="????",
    font_size=20,  # the size of emoji
    falling_speed=3,  # speed of raining
    animation_length="infinite",  # for how much time the animation will happen
)
  if polarity >= 0:
    st.success("The entered text has positive sentiments associated with it."+str(polarity))
    rain(
    emoji="????",
    font_size=20,  # the size of emoji
    falling_speed=3,  # speed of raining
    animation_length="infinite",  # for how much time the animation will happen
    )
  st.success(result)
