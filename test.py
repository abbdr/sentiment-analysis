import pandas as pd
import numpy as np
import KNN
from KNN import train_test_split

df = pd.read_csv('clean_tweets.csv')

X_train, y_train, X_test, y_test = train_test_split(df)


knn = KNN.KNN()
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)

def calculate_accuracy(predictions):
  prediction = []
  if(len(predictions.keys())==2):
    predictions = 'positive' if predictions['positive'] > predictions['negative'] else 'negative'
    prediction.append(predictions)
  elif(len(predictions.keys())==1):
    predictions = list(predictions.keys())[0]
    prediction.append(predictions)
  else:
    prediction.append(np.nan)

  acc = []
  for pred, real in zip(prediction, y_test):
    print(pred==real)
    acc.append(1 if pred==real else 0)

  accuracy = sum(acc) / len(acc)
  return accuracy
  
print(f'Accuracy is: {calculate_accuracy(predictions)} %')