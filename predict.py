import pandas as pd
import numpy as np
import KNN
from KNN import train_test_split

df = pd.read_csv('clean_tweets.csv')

X_train, y_train, X_test, y_test = train_test_split(df)

find_sentiment = input('Type something : ')

knn = KNN.KNN()
knn.fit(X_train, y_train)
predictions = knn.predict([find_sentiment])