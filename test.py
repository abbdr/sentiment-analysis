import pandas as pd
import numpy as np
import KNN
from KNN import train_test_split

df = pd.read_csv('clean_tweets.csv')

X_train, y_train, X_test, y_test = train_test_split(df)


knn = KNN.KNN()
knn.fit(X_train, y_train)
ready_test, predictions = knn.predict(X_test)

def calculate_accuracy(predictions):
  prediction = []
  for i in predictions:
    if(len(i.keys())==2):
      i = 'positive' if i['positive'] > i['negative'] else 'negative'
      prediction.append(i)
    elif(len(i.keys())==1):
      i = list(i.keys())[0]
      prediction.append(i)
    else:
      prediction.append(np.nan)

  acc = []
  for pred, real in zip(prediction, y_test):
    print(pred==real)
    acc.append(1 if pred==real else 0)

  accuracy = sum(acc) / len(acc)
  return accuracy
  
print(f'Accuracy is: {calculate_accuracy(predictions)} %')