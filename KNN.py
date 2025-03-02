import pandas as pd
import numpy as np
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

stop_words = set(stopwords.words('indonesian'))
factory = StemmerFactory()
stemmer = factory.create_stemmer()


def tf(terms, documents):
  tf = []
  for kalimat in documents:
    k = []
    for term in terms:
      p = [kata for kata in kalimat if kata==term]
      k.append(len(p)/len(kalimat))
    tf.append(k)
  tf = np.array(tf)
  return tf


def idf(length, terms, documents):
  tw = []
  for kalimat in documents:
    k = []
    for kata in terms:
      if kata in kalimat:
        k.append(1)
      else:
        k.append(0)
    tw.append(k)
  document_frequencies = np.array(tw).T
  # print('shape of document_frequencies: ', document_frequencies.shape)
  document_frequencies = np.array([np.sum(i) for i in document_frequencies])
  # print(np.log(length/document_frequencies[0]))
  return np.log(length/document_frequencies)


def tf_idf_(tf, idf):
  _tf_idf = []
  for every_document in tf:
    _tf_idf.append(every_document * idf)
  return np.array(_tf_idf)


def cos_similarity(tf_idf):
  x = tf_idf[:-1]
  y = tf_idf[-1]
  cs = np.array([np.dot(x[i], y) / (np.linalg.norm(x[i]) * np.linalg.norm(y)) for i in range(len(x))])
  return cs

def preprocessing(tweets):
    tweets = re.compile(r'https?://\S+|www\.\S+').sub(r'',tweets)
    tweets = re.compile(r'<.*?>').sub(r'',tweets)
    tweets = tweets.lower().split(' ')
    word = []
    for i in tweets:
      if re.match("^[a-z]+$", i):
        if i not in stop_words:
          i = stemmer.stem(i)
          word.append(i)
    word = [i for i in word if len(i)>3]
    return ' '.join(word)

def train_test_split(df, split_ratio=0.8):
  shuffle_df = df.sample(frac=1)
  shuffle_df = df

  # Define a size for your train set 
  train_size = int(0.8 * len(df))

  # Split your dataset 
  train_set = shuffle_df[:train_size]
  test_set = shuffle_df[train_size:]


  X_train = np.array(train_set['Clean Tweets'])
  y_train = np.array(train_set['Sentiments'])
  X_test = np.array(test_set['Clean Tweets'])
  y_test = np.array(train_set['Sentiments'])
  # print(y_train)

  return X_train, y_train, X_test, y_test


class KNN:
  def __init__(self, k=3):
      self.k = k

  def fit(self, X, y):
    self.X_train = np.array(X)
    self.y_train = np.array(y)

  def predict(self, X):
    if not (str(type(X))=="<class 'numpy.ndarray'>" or str(type(X))=="<class 'list'>"):
      print('Type of X_test must be array')
      return

    predictions = []
    for sentence in X:
      if not str(type(sentence))=="<class 'str'>":
        print('Type of each element in X_test must be string')
        return 
      
      # print(find);return
      find = preprocessing(sentence)

      tweets = list(self.X_train)
      tweets.append(find)
      terms = np.array(sorted(set([j for i in tweets for j in i.split(' ')])))
      tweets = [i.split(' ') for i in tweets ]

      term_frequencies = tf(terms=terms, documents=tweets)
      inverse_document_frequencies = idf(length = len(terms), terms=terms, documents=tweets)
      tf_idf = tf_idf_(tf=term_frequencies, idf=inverse_document_frequencies)
      cosine_similarity = cos_similarity(tf_idf)

      ready_test = pd.concat({'Tweets' : pd.Series(cosine_similarity), 'Sentiment' : pd.Series(self.y_train)}, axis=1)
      ready_test = ready_test.query('Tweets > 0')
      ready_test = ready_test.sort_values(by=['Tweets'], ascending=False)
      ready_test

      result = np.array(ready_test['Sentiment'][:self.k])
      unique, counts = np.unique(result, return_counts=True)
      result_count = dict(zip(unique, list(counts)))
      print(f'{result_count} -> {sentence}')

      predictions.append(result_count)
  
    return predictions

  
