#!/usr/bin/python3

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import joblib

enron_data = joblib.load(open("../final_project/final_project_dataset_unix.pkl", "rb"))

# num of data points
length = len(enron_data)
print("Length of dataset: %d" % length)

# num of features (METTS MARK is a person in dataset)
featuresLength = len(enron_data["METTS MARK"])
print("Features Length: %d" % featuresLength)

# num of person of interests in dataset
counter = 0
for i in enron_data:
    if enron_data[i]["poi"] == 1:
        counter += 1
print("Persons of interests: %d" % counter)

# persons of interest in the names file
POItext = open("../final_project/poi_names.txt", "rb")
nameCount = 0
for i in POItext:
    nameCount += 1
nameCount -= 2
print("Number of POIS in text file: %d" % nameCount)

# stock for James Prentice
Pstock = enron_data["PRENTICE JAMES"]["total_stock_value"]
print("James Prentice's stock: %d" % Pstock)

# emails to POI for Colwell
Cemail = enron_data["COLWELL WESLEY"]["from_this_person_to_poi"]
print("Wesley Colwell's emails to POIs: %d" % Cemail)

# stock options of Jeff Skilling
Sstock = enron_data["SKILLING JEFFREY K"]["exercised_stock_options"]
print("Jeff Skilling's stock options: %d" % Sstock)

# individuals with most money
Lmoney = enron_data["LAY KENNETH L"]["total_payments"]
Smoney = enron_data["SKILLING JEFFREY K"]["total_payments"]
Fmoney = enron_data["FASTOW ANDREW S"]["total_payments"]

print("Money: %d, %d, %d" % (Lmoney, Smoney, Fmoney))

# people with salary and known email address
salaryCount = 0
emailCount = 0
for i in enron_data:
    if enron_data[i]["salary"] != "NaN":
        salaryCount += 1
    if enron_data[i]["email_address"] != "NaN":
        emailCount += 1

print("Known salary: %d" % salaryCount)
print("Known email addresses: %d" % emailCount)

# people with no payments
paymentCount = 0

for i in enron_data:
    if enron_data[i]["total_payments"] == "NaN":
        paymentCount += 1

paymentPer = paymentCount / length * 100
print("Percentage of people missing payment count: %f%%" % paymentPer)

# POI with no payments
POIpaymentCount = 0

for i in enron_data:
    if enron_data[i]["poi"] == 1:
        if enron_data[i]["total_payments"] == "NaN":
            POIpaymentCount += 1

POIpaymentPer = POIpaymentCount / counter * 100
print("Percentage of POIs missing payment count: %f%%" % POIpaymentPer)