import pandas as pd
import numpy as np
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
from nltk.corpus import stopwords
# nltk.download('stopwords')
from KNN import preprocessing
import streamlit as st

stop_words = set(stopwords.words('indonesian'))
factory = StemmerFactory()
stemmer = factory.create_stemmer()

'## Sentiment Analysis Menggunakan Dataset Pilkada DKI Jakarta'

df = pd.read_csv('dataset_tweet_sentiment_pilkada_DKI_2017.csv')
'#### Dataset Asli'
df

negative = df[df['Sentiment'] == 'negative']
'#### Negative values'
negative

positive = df[df['Sentiment'] == 'positive']
'#### Positive values'
positive


post = pd.read_csv('clean_tweets.csv')

'#'
'### Tahapan Preprocessing'
tweets = ''


'#### 1. Menghapus link atau url'
with st.echo():
  re.compile(r'https?://\S+|www\.\S+').sub(r'',tweets)

  
'#### 2. Menghapus karakter html'
with st.echo():
  re.compile(r'<.*?>').sub(r'',tweets)

  
'#### 3. Mengubah semua kalimat menjadi huruf kecil kemudian menjadikannya array yang berisi tiap kata dari kalimat'
with st.echo():
  tweets.lower().split(' ')


'#### 4. Memfilter kata yang bukan alfabet'
with st.echo():
  for i in tweets:
    re.match("^[a-z]+$", i)

  
'#### 5. Membuang kata yang termasuk stopword (kata yang tidak memiliki makna)'
with st.echo():
  for i in tweets:
    if i in stop_words:
      'buang'


'#### 5. Membuat semua kata menjadi kata dasar'
with st.echo():
  for i in tweets:
     i = stemmer.stem(i)


'#'
'### Setelah Preprocessing'
post

st.page_link('pages/2-classify.py', label='klasifikasi')