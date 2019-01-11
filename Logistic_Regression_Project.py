

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



# Get the Data

ad_data = pd.read_csv('advertising.csv')


# Check the head of ad_data

print(ad_data.head())


# Use info and describe
print(ad_data.info())

print(ad_data.describe())


#  Exploratory Data Analysis
# Create a histogram of the Age

sns.set_style('whitegrid') # just to make the chart a little easier to read
ad_data['Age'].hist(bins=30)
plt.xlabel('Age')


# Create a jointplot showing Area Income versus Age


print(sns.jointplot(x='Age',y='Area Income',data=ad_data))


# Create a jointplot showing the kde distributions of Daily Time spent on site vs. Age.


print(sns.jointplot(x='Age',y='Daily Time Spent on Site',data=ad_data,color='red',kind='kde'))


# Create a jointplot of 'Daily Time Spent on Site' vs. 'Daily Internet Usage'


print(sns.jointplot(x='Daily Time Spent on Site',y='Daily Internet Usage',data=ad_data,color='green'))


# a pairplot with the hue defined by the 'Clicked on Ad' column feature.

print(sns.pairplot(ad_data,hue='Clicked on Ad',palette='bwr'))


# Actual Logistic Regression part here

from sklearn.model_selection import train_test_split

# Split the data into training set and testing set using train_test_split

X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Male']]
y = ad_data['Clicked on Ad']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# Train and fit a logistic regression model on the training set.

from sklearn.linear_model import LogisticRegression


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# ## Predictions and Evaluations
# Now predicting values for the testing data.

predictions = logmodel.predict(X_test)


# Create a classification report and confusion matrix from the model.


from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test,predictions))

print(confusion_matrix(y_test,predictions))

