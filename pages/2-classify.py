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
  predictions, ready_test = knn.predict([find_sentiment])
  ready_test
  '#'
  '#### Hasil: '
  predictions[0]
  '#'
  if(len(predictions.keys())==2):
    '#### prediction: positive' if predictions['positive'] > predictions['negative'] else '#### prediction: negative'
  elif(len(predictions.keys())==1):
    f'#### prediction: {list(predictions.keys())[0]}'
  else:
    '#### prediction: NaN'

