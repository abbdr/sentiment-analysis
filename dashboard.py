import streamlit as st
import pandas as pd
import numpy as np
import KNN
from KNN import train_test_split

post = pd.read_csv('clean_tweets.csv')

X_train, y_train, X_test, y_test = train_test_split(post)

'## Sentiment Analysis Menggunakan Dataset Pilkada DKI Jakarta'

'''
#### Menggunakan ALgoritma K-Nearest Neighbor
#### Dihitung menggunakan Cosine Similarity, salah satu metode untuk mengetahui kemiripan antarkalimat
#### Setiap kalimat diberikan pembobotan dengan menggunakan metode TF IDF (Terms Frequeny - Inverse Document Frequency)
#
'''

st.page_link('pages/1-preprocessing.py', label='preprocessing')
