#!/usr/bin/python3

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time

sys.path.append("../tools/")
from email_preprocess import preprocess

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###

from sklearn.svm import SVC

# shorten training dataset
# features_train = features_train[:int(len(features_train)/100)]
# labels_train = labels_train[:int(len(labels_train)/100)]

# initialize SVM with linear decision boundary
clf = SVC(C = 10000, kernel = "rbf")

# fit to data
clf.fit(features_train, labels_train)

# finding predicted labels for test points
pred = clf.predict(features_test)

# getting accuracy
from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)

print("Accuracy: %f" % acc)

# time for training
t0 = time()
clf.fit(features_train, labels_train)
print("Training time:", round(time() - t0, 3), "s")

# time for predicting
t0 = time()
clf.predict(features_test)
print("Predicting time:", round(time() - t0, 3), "s")

# Predicting specific index
answer1 = pred[10]
answer2 = pred[26]
answer3 = pred[50]
print(answer1, answer2, answer3)

# Number of Chris:
counter = 0
for i in pred:
    if i == 1:
        counter += 1

print("Number of Chris emails: %d" % counter)

#########################################################

#########################################################
'''
You'll be Provided similar code in the Quiz
But the Code provided in Quiz has an Indexing issue
The Code Below solves that issue, So use this one
'''

# features_train = features_train[:int(len(features_train)/100)]
# labels_train = labels_train[:int(len(labels_train)/100)]

#########################################################
