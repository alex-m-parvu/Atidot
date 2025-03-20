**Churn Prediction Script**
==========================

A Python script designed to predict customer churn using machine learning.

**Table of Contents**
-----------------

1. [Introduction](#introduction)
2. [Requirements](#requirements)
3. [Script Description](#script-description)
4. [Step-by-Step Explanation](#step-by-step-explanation)

**Introduction**
---------------

This script uses a random forest classifier to predict customer churn based on various features such as usage patterns, demographics, and billing information.

**Requirements**
--------------

* Python 3.x
* pandas
* scikit-learn
* matplotlib
* numpy
* joblib

**Script Description**
--------------------

The churn prediction script performs the following tasks:

1. Loads the dataset from a CSV file.
2. Splits the data into training and testing sets.
3. Trains a random forest classifier on the training set.
4. Evaluates the performance of the trained model using precision, recall, and classification report metrics.
5. Performs feature importance analysis to identify the most relevant features for churn prediction.

**Step-by-Step Explanation**
---------------------------

### Step 1: Data Loading

The script starts by loading the dataset from a CSV file named `churn_data.csv`. The dataset contains various features related to customer behavior, demographics, and billing information. The script uses pandas to read the CSV file 
into a DataFrame object.

```python
import pandas as pd
data = pd.read_csv('churn_data.csv')
```

### Step 2: Data Splitting

The script splits the loaded data into training and testing sets using a stratified split, which preserves the class balance in both sets. The train_test_split function from scikit-learn is used to achieve this.

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data.drop('churn', axis=1), data['churn'], test_size=0.2, random_state=42)
```

### Step 3: Random Forest Classifier Training

The script trains a random forest classifier on the training set using the RandomForestClassifier class from scikit-learn. The classifier is trained with default parameters.

```python
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)
```

### Step 4: Model Evaluation

The script evaluates the performance of the trained random forest classifier using precision, recall, and classification report metrics from scikit-learn.

```python
from sklearn.metrics import precision_score, recall_score, classification_report
y_pred = rfc.predict(X_test)
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
```

### Step 5: Feature Importance Analysis

The script performs feature importance analysis to identify the most relevant features for churn prediction using the feature_importances_ attribute of the random forest classifier.

```python
import matplotlib.pyplot as plt
feature_importances = rfc.feature_importances_
plt.bar(range(len(feature_importances)), feature_importances)
plt.xlabel("Feature Index")
plt.ylabel("Importance Score")
plt.title("Feature Importance Analysis")
plt.show()
```


