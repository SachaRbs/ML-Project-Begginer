# import main data analysis libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# note we use scipy for generating a uniform distribution in the model optimization step
from scipy.stats import uniform

# note that because of the different dataset and algorithms, we use different sklearn libraries from Day 1 
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.dummy import DummyClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

# hide warnings
import warnings
warnings.filterwarnings('ignore')

def evaluate(y_test, y_pred):
    # this block of code returns all the metrics we are interested in 
    accuracy = metrics.accuracy_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    auc = metrics.roc_auc_score(y_test, y_pred)

    print ("Accuracy", accuracy)
    print ('F1 score: ', f1)
    print ('ROC_AUC: ' , auc)

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

full_dataset = X.copy()
full_dataset['target'] = y.copy()

# sns.pairplot(full_dataset)
# plt.figure(figsize= (15, 10))
# sns.heatmap(full_dataset.corr(method='pearson'))
# plt.show()

xtrain, xtest, ytrain, ytest = train_test_split(X, y, train_size=0.8 , stratify = y, shuffle=True)

#DummyClssifier:

    # model = DummyClassifier(strategy='uniform', random_state=1)
    # model.fit(xtrain, ytrain)
    # y_pred = model.predict(xtest)
    # evaluate(ytest, y_pred)

#LogisticRegression using cross_val_score:
logistic = LogisticRegression()
cross_val_score(logistic, X, y, cv=5, scoring="accuracy")

#RandomForestClassifier
rnd_cld = RandomForestClassifier()
cross_val_score(rnd_cld, X, y, cv=5, scoring='accuracy')

#DecisionTreeClassifier
dt_clf = DecisionTreeClassifier()
cross_val_score(dt_clf, X, y, cv=5, scoring='accuracy')

voting_clf = VotingClassifier(estimators=[('lr', logistic), ('rf', rnd_cld), ('dc', dt_clf)], voting='soft')

for clf in (logistic, rnd_cld, dt_clf, voting_clf):
    clf.fit(xtrain, ytrain)
    y_pred = clf.predict(xtest)
    print(clf.__class__.__name__, accuracy_score(ytest, y_pred))

#optimisation: Logistic Regression

penalty = ['l1', 'l2']

C = uniform(loc=0, scale=4)
param = dict(C=C, penalty=penalty)
rsv = RandomizedSearchCV(logistic, param, random_state=1, n_iter=100, cv=5, verbose=0, n_jobs=-1)
bestmodel = rsv.fit(X, y)
print(bestmodel.best_estimator_)
print(cross_val_score(rsv, X, y, cv=5, scoring="accuracy").mean())