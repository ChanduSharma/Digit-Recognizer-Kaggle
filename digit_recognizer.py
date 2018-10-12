# Digit Recognizer

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the Training
dataset = pd.read_csv("train.csv")
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values

X[X>0] = 1

# Visualizing the first image from the dataset
i = 6
first_image = X[i].reshape((28,28))
plt.imshow(first_image, cmap='binary')
plt.title('Image of {}'.format(y[i]))


# Splitting the dataset into Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting the Support vector classifier to the training set
from sklearn.svm import SVC
clf = SVC()
clf.fit(X_train, y_train)

# Predicting the output for the test set
y_pred = clf.predict(X_test)

# Making the confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
clf.score(X_test, y_test)

# Reading the test dataset
test_set = pd.read_csv('test.csv')
test_set[test_set > 0] = 1
# Submission for the kaggle competition
results = clf.predict(test_set)
df = pd.DataFrame(results)
df.index.name='ImageId'
df.index+=1
df.columns=['Label']
df.to_csv('results.csv', header=True)
