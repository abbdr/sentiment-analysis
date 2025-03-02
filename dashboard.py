import streamlit as st
import pandas as pd
import numpy as np
import KNN
from KNN import train_test_split

post = pd.read_csv('dataset_tweet_sentiment_pilkada_DKI_2017.csv')


'## Sentiment Analysis Menggunakan Dataset Pilkada DKI Jakarta'

'''
#### Menggunakan Algoritma K-Nearest Neighbor
#### Dihitung menggunakan Cosine Similarity, salah satu metode untuk mengetahui kemiripan antarkalimat
#### Setiap kalimat diberikan pembobotan dengan menggunakan metode TF IDF (Terms Frequeny - Inverse Document Frequency)
'''

post


st.page_link('pages/1-preprocessing.py', label='preprocessing')
