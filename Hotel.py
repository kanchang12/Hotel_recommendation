import nltk
#nltk.download('wordnet')
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from ast import literal_eval

dataset = pd.read_csv("Hotel_Reviews.csv")

#print(dataset.head())

dataset.Hotel_Address = dataset.Hotel_Address.str.replace("United Kingdom", "UK")
dataset["Country"] = dataset.Hotel_Address.apply(lambda x: x.split(' ')[-1])
print(dataset.Country.unique())

dataset.drop(['Additional_Number_of_Scoring', 'Review_Date','Reviewer_Nationality','Negative_Review', 'Review_Total_Negative_Word_Counts','Total_Number_of_Reviews', 'Positive_Review','Review_Total_Positive_Word_Counts','Total_Number_of_Reviews_Reviewer_Has_Given', 'Reviewer_Score','days_since_review', 'lat', 'lng'], axis=1, inplace=True)

#print(dataset.head())

def impute(column):
    column = column[0]
    if (type(column)!= list):
        return "".join(literal_eval(column))
    else:
        return column

dataset["Tags"]= dataset[["Tags"]].apply(impute, axis =1)
#print(dataset.head())

dataset["Country"] = dataset["Country"].str.lower()
dataset["Tags"] = dataset["Tags"].str.lower()


def recommend(Country, description):
    description = description.lower()
    description = word_tokenize(description)  # Fix: Tokenize the description
    stop_words = stopwords.words('english')
    lemm = WordNetLemmatizer()
    filtered  = {word for word in description if not word in stop_words}
    filtered_set = set()
    for fs in filtered:
        filtered_set.add(lemm.lemmatize(fs))

    country = dataset[dataset['Country']==Country.lower()]
    country = country.set_index(np.arange(country.shape[0]))
    list1 = []; list2 = []; cos = [];
    for i in range(country.shape[0]):
        temp_token = word_tokenize(country["Tags"][i])
        temp_set = [word for word in temp_token if not word in stop_words]
        temp2_set = set()
        for s in temp_set:
            temp2_set.add(lemm.lemmatize(s))
        vector = temp2_set.intersection(filtered_set)
        cos.append(len(vector))
    country['similarity']=cos
    country = country.sort_values(by='similarity', ascending=False)
    country.drop_duplicates(subset='Hotel_Name', keep='first', inplace=True)
    country.sort_values('Average_Score', ascending=False, inplace=True)
    country.reset_index(inplace=True)
    print(country[["Hotel_Name", "Average_Score", "Hotel_Address"]].head())
place = input("Choose Between Netherlands' 'UK' 'France' 'Spain' 'Italy' 'Austria': ")
purpose = input("I am going on a ")
purpose = "I am going on a " + purpose
recommend(place, purpose)