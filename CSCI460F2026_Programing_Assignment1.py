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
import numpy as np




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


x = df.drop(columns = ["y"])
y = df["y"]

print("")
print("")
print("")
print("x is : ")
print(x)

print("")
print("")
print("")
print(y)

print("")
print("")
print("")
print("Class proportions:\n", y.value_counts(normalize=True))
print("")
print("")
print("")

#now training part
#x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, stratify = y)
#model = GaussianNB()
#model.fit(x_train, y_train)
#y_prediction = model.predict(x_test)

#accuracy = accuracy_score(y_prediction, y_test)
#f1 = f1_score(y_prediction, y_test, average = "weighted")
#print("The accuracy of Naive Bayes Model is: ", accuracy)
#print("The F1 score of Naive bayes Model is: ", f1)


def doTrial(train_s):
	x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = train_s, stratify = y)
	model = GaussianNB()
	model.fit(x_train, y_train)
	y_prediction = model.predict(x_test)

	accuracy = accuracy_score(y_prediction, y_test)
	f1 = f1_score(y_prediction, y_test, average = "weighted")
#	print("The accuracy of Naive Bayes Model is: ", accuracy)
#	print("The F1 score of Naive bayes Model is: ", f1)
	return accuracy, f1

#------task 2 -----------------

accuracies = []
f1s = []

numTrials = 10
#numTrials = int(input("How many number of trials? (10): "))
print("Trial    Accuracy    F1-Score")
for trial in range(1, numTrials+1):
	a,f = doTrial(0.7)
	accuracies.append(a)
	f1s.append(f)
	print(trial,"     ", round(a,4), "     ", round(f,4))

print("mean accuracies: ", round(np.mean(accuracies), 4),"mean f1s: ", round(np.mean(f1s), 4))
print("Standard deviation accuracies: ", round(np.std(accuracies), 4), "Standard deviation f1s: ", round(np.std(f1s), 4))

#------------Task 3_______________
#run program 9 times with 10 iterations each

#10% training and 90% testing for first and move up 10% training and down 10% testing

print("Train%    Test%    AccuracyMean    AccuracyStd    F1Mean    F1Std")


for i in range(10, 100, 10):
	frac = i /100
	accuracyList = []
	f1list = []

	for trial in range(1, 11):
		a, f = doTrial(frac)
		accuracyList.append(a)
		f1list.append(f)

	print(i, "       ", 100 - i, "    ", round(np.mean(accuracyList), 4), "         ", round(np.std(accuracyList), 4), "     ", round(np.mean(f1list), 4), "   ", round(np.std(f1list), 4))






























