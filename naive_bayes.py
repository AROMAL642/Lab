import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

# Load the dataset
ds = pd.read_csv('/content/heart.csv')

# Split the data into features and target variable
x = ds.drop(columns='target', axis=1)
y = ds['target']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# Print the shapes of the datasets
print(x.shape, x_train.shape, x_test.shape)

# Initialize and train the Naive Bayes model
nb = GaussianNB()
nb.fit(x_train, y_train)

# Evaluate the model on the training set
training_prediction = nb.predict(x_train)
training_accuracy = metrics.accuracy_score(training_prediction, y_train)
print('Training accuracy =', training_accuracy)

# Evaluate the model on the testing set
testing_prediction = nb.predict(x_test)
testing_accuracy = metrics.accuracy_score(testing_prediction, y_test)
print('Testing accuracy =', testing_accuracy)

# Test the model with a single input
input_data = (62, 1, 0, 120, 267, 0, 1, 99, 1, 1.8, 1, 2, 3)
input_array = np.asarray(input_data)
reshape_array = input_array.reshape(1, -1)
prediction = nb.predict(reshape_array)

# Interpret the prediction
if prediction[0] == 0:
    print('Prediction: No')
else:
    print('Prediction: Yes')
