import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()

X, y = iris.data, iris.target

xtrain, xtest, ytrain, ytest = train_test_split(X, y, train_size=0.8)

bestk = 1
best_accuracy = 0
for k in range(1, 30):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(xtrain, ytrain)
    y_hat = knn.predict(xtest)
    accuracy = accuracy_score(ytest, y_hat)
    print(k, accuracy)
    if accuracy > best_accuracy:
        best_model = knn
        best_accuracy = accuracy
