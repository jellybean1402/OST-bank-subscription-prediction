#!/bin/bash

import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, jaccard_score, roc_auc_score, precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Define paths
DATA_DIR = "./data"
MODEL_DIR = "./models"

# Load new data
data_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.xlsx')]
latest_file = max(data_files, key=lambda f: os.path.getctime(os.path.join(DATA_DIR, f)))

print(f"Using the latest data file: {latest_file}")
df = pd.read_excel(os.path.join(DATA_DIR, latest_file))
df.y = df.y.map({'no': 0, 'yes': 1})

# Splitting the data into train, validation and test datasets
train, val, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])

Xtrain = train.iloc[:, :-1]
ytrain = train.y
Xval = val.iloc[:, :-1]
yval = val.y
Xtest = test.iloc[:, :-1]
ytest = test.y

Xtrain.rename(columns= {'job_blue-collar': 'job_blue_collar', 'job_self-employed': 'job_self_employed'}, inplace=True)
Xval.rename(columns= {'job_blue-collar': 'job_blue_collar', 'job_self-employed': 'job_self_employed'}, inplace=True)
Xtest.rename(columns= {'job_blue-collar': 'job_blue_collar', 'job_self-employed': 'job_self_employed'}, inplace=True)

#Encoding the data
encoding_columns = list(df.iloc[:,:-1].select_dtypes(include=object))

education_status = {'unknown': 0,
       'primary': 1,
       'secondary': 2,
       'tertiary': 3}

# Encoding the education attribute
Xtrain['education'] = Xtrain['education'].replace(education_status)
Xval['education'] = Xval['education'].replace(education_status)
Xtest['education'] = Xtest['education'].replace(education_status)

# One Hot Encoding using Pandas
ohe_cols = list(df.iloc[:,:-1].select_dtypes(include=object))

# Performing One Hot on columns
Xtrain = pd.get_dummies(Xtrain ,columns = ohe_cols ,prefix=ohe_cols)
Xval = pd.get_dummies(Xval ,columns = ohe_cols ,prefix=ohe_cols)
Xtest = pd.get_dummies(Xtest ,columns = ohe_cols ,prefix=ohe_cols)

Xtrain.rename(columns= {'job_admin.': 'job_admin','job_blue-collar': 'job_blue_collar', 'job_self-employed': 'job_self_employed'}, inplace=True)
Xval.rename(columns= {'job_admin.': 'job_admin','job_blue-collar': 'job_blue_collar', 'job_self-employed': 'job_self_employed'}, inplace=True)
Xtest.rename(columns= {'job_admin.': 'job_admin','job_blue-collar': 'job_blue_collar', 'job_self-employed': 'job_self_employed'}, inplace=True)

# Train model
model = RandomForestClassifier(random_state = 42)

model.fit(Xtrain, ytrain)

test_prediction = model.predict(Xtest)

# Evaluate (placeholder, you can expand this)
accuracy = accuracy_score(ytest, test_prediction)
print(f"Model accuracy: {accuracy}")

# Save model with timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs(MODEL_DIR, exist_ok=True)
model_filename = f"model_{timestamp}.pkl"
model_path = os.path.join(MODEL_DIR, model_filename)
joblib.dump(model, model_path)

print(f"Model saved as: {model_path}")
