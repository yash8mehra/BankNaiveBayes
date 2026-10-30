######################################################

#  Student Name: <Yash Mehra>

#  Course Code: CSCI 460 -- Fall 2026

#  Assignment Due Date: <10/25/26>

#  GitHub Link: <https://github.com/yash8mehra/BankNaiveBayes>

######################################################


######################################################
#             Write a Python program to do the following:
#  Load the above dataset file into a data frame and call it df
#  Map each categorical variables into an integer number starting at 0. This applies to the
#  columns: ‘job’, 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month',
#  'poutcome', and the target column ‘y’ (see the mapping example in the hints below)
#  Remove any rows that have missing (NA) values from df
#  Set X = all feature columns in the data frame excluding the target column y
#  Set y = the target column ‘y’
######################################################

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy




df = pd.read_csv("bank-full.csv", sep=";")
print("The shape is: ", df.shape)

#categorial columns only
columns = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome", "y"]

for i in columns:
	labels = sorted(df[i].dropna().unique())
	d = {}
	num = 0
	for label in labels:
		d[label] = num
		num += 1

	print("Mapping for", i, ":", d)
	df[i] = df[i].map(d)





