#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""
import os
import joblib
import sys
sys.path.append(os.path.abspath("../tools/"))
from feature_format import featureFormat, targetFeatureSplit

data_dict = joblib.load(open("../final_project/final_project_dataset_unix.pkl", "rb") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list, sort_keys = '../tools/python2_lesson14_keys_unix.pkl')
labels, features = targetFeatureSplit(data)

### your code goes here 

from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, random_state = 42, test_size = 0.3)

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test, labels_test)
# print(clf.score(features_test, labels_test))

# get total number of POIs (1s) in test set
POIcounter = 0

for i in labels_test:
    if i == 1:
        POIcounter += 1
print(POIcounter)

print(len(labels_test))

# get total number of POIs that are correctly predicted in test set
truePos = 0

for i in range(len(labels_test)):
    if pred[i] == 1 and labels_test[i] == 1:
        truePos += 1

print(truePos)

# Precision and Recall
from sklearn.metrics import precision_score, recall_score

print(precision_score(y_true = labels_test, y_pred = pred))

print(recall_score(y_true = labels_test, y_pred = pred))