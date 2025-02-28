import streamlit as st
import pandas as pd
import numpy as np
import KNN
from KNN import train_test_split

post = pd.read_csv('clean_tweets.csv')

X_train, y_train, X_test, y_test = train_test_split(post)

'### Tulis Sesuatu Tentang Pilkada DKI Jakarta: '
find_sentiment = st.text_input('')

if st.button('Analyze'):
  knn = KNN.KNN()
  knn.fit(X_train, y_train)
  predictions = knn.predict([find_sentiment])[0]
  '#### Hasil: '
  predictions
  '#'
  if(len(predictions.keys())==2):
    '#### positive' if predictions['positive'] > predictions['negative'] else '#### negative'
  elif(len(predictions.keys())==1):
    f'#### {list(predictions.keys())[0]}'
  else:
    '#### NaN'

