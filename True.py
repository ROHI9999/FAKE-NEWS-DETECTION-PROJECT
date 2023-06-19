import csv
from turtle import done
import requests
import json
import numpy as np
import pandas as pd
import itertools
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from requests.api import head

# This code fetches the top headlines from the news API and saves them to a CSV file. 
import csv 
import requests 

url = "https://newsapi.org/v2/top-headlines?country=in&apiKey=a5c316260e39428d9d07fd53020551c3" 

headers = {
    'Accept': 'application/json',
    'Content-Type': 'application/json',
}
response = requests.request("GET", url, headers=headers, data={})
myjson = response.json()
ourdata = []
csvheader = ['AUTHOR', 'TITLE', 'PUBLISHED AT', 'LABEL']

for x in myjson['articles']:
    author = x['author'] if x['author'] else 'Anonymous'
    title = x['title']
    published_at = x['publishedAt']
    label = 'true' # Adding a label of "true" to each article
    listing = [author, title, published_at, label]
    ourdata.append(listing)
    ourdata.append([])

with open('true.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(csvheader)
    writer.writerows(ourdata)








# with open('new.csv', 'a', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow([])






# # Read the CSV data into a list
# with open('new.csv', 'r') as file:
#     data = list(csv.reader(file))

# # Add new data to odd rows starting from the 3rd row
# new_data = ['new_value1', 'new_value2', 'new_value3']
# for i in range(2, len(data), 2):
#     data[i] = new_data

# # Write the modified data back to the CSV file
# with open('new.csv', 'w', newline='') as file:
#     csv.writer(file).writerows(data)




# import csv

# filename = "new.csv"

# with open(filename, "r") as f:
#     data = list(csv.reader(f))

# for i in range(1, len(data)):
#     data[i].append("TRUE" if i % 2 == 1 else "FALSE")

# with open(filename, "w", newline='') as f:
#     csv.writer(f).writerows(data)






# # Read the data
# df=pd.read_csv("new.csv")
# print(df)
# # Get shape and head
# print(df.head())
# print(df.shape)

# #DataFlair - Get the labels
# titles=df.TITLE
# print(titles.head())

# #DataFlair - Get the labels
# labels=df.label
# labels.head()


# #DataFlair - Split the datase
# x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.2, random_state=7)

# #DataFlair - Initialize a TfidfVectorizer
# tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)

# #DataFlair - Fit and transform train set, transform test set
# tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
# tfidf_test=tfidf_vectorizer.transform(x_test)

# #DataFlair - Initialize a PassiveAggressiveClassifier
# pac=PassiveAggressiveClassifier(max_iter=50)
# pac.fit(tfidf_train,y_train)

# #DataFlair - Predict on the test set and calculate accuracy
# y_pred=pac.predict(tfidf_test)
# score=accuracy_score(y_test,y_pred)
# print(f'Accuracy: {round(score*100,2)}%')

# #DataFlair - Build confusion matrix
# confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])