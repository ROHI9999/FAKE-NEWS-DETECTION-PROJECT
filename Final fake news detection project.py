# Importing necessary libraries
import pandas as pd 
import numpy as np 
import re
from sklearn.model_selection import train_test_split as ttp 
from sklearn.metrics import classification_report 
import string 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.ensemble import RandomForestClassifier


# Reading datasets from the csv files using pandas library
data_true = pd.read_csv("True.csv") 
data_fake = pd.read_csv("Fake.csv") 


# Adding class column based on the label of each dataset 
data_true["class"] = True           
data_fake["class"] = False


# Merging the true and fake data sets into a single set 
data_merge = pd.concat([data_fake, data_true], axis=0) 


# Defining filtering function to filter Title of news articles 
def filtering(data):                                   
    TITLE = data.lower()
    TITLE = re.sub('\[.*?\]', '', TITLE)
    TITLE = re.sub("\W", " ", TITLE) 
    TITLE = re.sub('https?://\S+|www.\S+', '', TITLE) 
    TITLE = re.sub('<.*?>+', '', TITLE)
    TITLE = re.sub('[%s]' % re.escape(string.punctuation), '', TITLE) 
    TITLE = re.sub('\w*\d\w*', '', TITLE) 
    return TITLE


# Filtering the data in "TITLE" column using filter function defined above
data_merge["TITLE"] = data_merge["TITLE"].apply(filtering)


# Segregating the filtered titles and assigned class as "FAKE" or "TRUE"
x = data_merge["TITLE"] 
y = data_merge["class"] 


# Splitting the dataset into training set and test set
x_train, x_test, y_train, y_test = ttp(x, y, test_size=0.25, random_state=0) 


# Creating features named XV matrices for train and test datasets
vector = TfidfVectorizer()
xv_train = vector.fit_transform(x_train)
xv_test = vector.transform(x_test)


# Building a Random Forest Classifier model
RFC = RandomForestClassifier(random_state=0)     
RFC.fit(xv_train, y_train)


# Generating Classification metrics report using above model
pred_RFC = RFC.predict(xv_test)
print(classification_report(y_test, pred_RFC))


# Building a function named manual_testing to manually test our model by entering news articles
def manual_testing(news):
    testing_news = {"TITLE": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["TITLE"] = new_def_test["TITLE"].apply(filtering)
    new_x_test = new_def_test["TITLE"]
    new_xv_test = vector.transform(new_x_test)
    pred_RFC = RFC.predict(new_xv_test)

    return print(pred_RFC)
    

# Taking user input to manually test our model   
news1 = input("Enter news 1: ")
result1 = manual_testing(news1)

news2 = input("Enter news 2: ")
result2 = manual_testing(news2)



# Uncomment the below line if you want to check your model's prediction on a particular news directly  
# manual_testing("Donald Trump wins the US Presidential Election by a Landslide")
