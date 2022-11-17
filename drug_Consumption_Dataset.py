import pandas as pd
import numpy as np
import pickle
from sklearn import tree, ensemble, svm, neighbors, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
respondents = pd.read_csv("drug_consumption.csv")

feature_list = [
"ID",
"Age",
"Gender",
"Education",
"Country",
"Ethnicity",
"Nscore",
"Escore",
"Oscore",
"Ascore",
"Cscore",
"Impulsive",
"SS",
"Alcohol",
"Amphet",
"Amyl",
"Benzos",
"Caff",
"Cannabis",
"Choc",
"Coke",
"Crack",
"Ecstasy",
"Heroin",
"Ketamine",
"Legalh",
"LSD",
"Meth",
"Mushrooms",
"Nicotine",
"Semer",
"VSA"
]
drugs_list = ["Alcohol",
"Amphet",
"Amyl",
"Benzos",
"Caff",
"Cannabis",
"Choc",
"Coke",
"Crack",
"Ecstasy",
"Heroin",
"Ketamine",
"Legalh",
"LSD",
"Meth",
"Mushrooms",
"Nicotine",
"Semer",
"VSA"]

# create features
respondents.columns = feature_list

# create labels
drugs = respondents[drugs_list].copy()
respondents = respondents.drop(drugs_list, axis=1)

# create label values function 
def BinaryLabel(df, column):
	for i in df[column].index:
		if(df[column][i]=="CL0" or df[column][i]=="CL1"):
			df[column][i] = 0
		else:
			df[column][i] = 1




# label data with value 1 for user and value 0 for non-user
for drug in drugs:
	BinaryLabel(drugs,drug)

# feature selection
threshold = 0.03
columns_to_drop = []
for column in respondents.columns:
	if(respondents[column].var()<threshold):
		columns_to_drop.append(column)
print(columns_to_drop)
respondents=respondents.drop(columns=columns_to_drop)


# use holdout method of 33% for creating training and testing set
respondents_train, respondents_test, drugs_train, drugs_test = train_test_split(respondents, drugs, test_size = 0.33)

with open('./data/respondents_train.pickle', 'wb') as f:
	pickle.dump(respondents_train, f)

with open('./data/respondents_test.pickle', 'wb') as f:
	pickle.dump(respondents_test, f)

with open('./data/drugs_train.pickle', 'wb') as f:
	pickle.dump(drugs_train, f)

with open('./data/drugs_test.pickle', 'wb') as f:
	pickle.dump(drugs_test, f)
