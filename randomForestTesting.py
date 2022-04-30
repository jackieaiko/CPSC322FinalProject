
from mysklearn.myclassifiers import MyKNeighborsClassifier, MyDummyClassifier, MyNaiveBayesClassifier, MyDecisionTreeClassifier, MyRandomForestClassifier
import mysklearn.myevaluation as myevaluation
from mysklearn import myutils
import numpy as np
import random
random.seed(0)
X = [
    ["Senior", "Java", "no", "no"],
    ["Senior", "Java", "no", "yes"],
    ["Mid", "Python", "no", "no"],
    ["Junior", "Python", "no", "no"],
    ["Junior", "R", "yes", "no"],
    ["Junior", "R", "yes", "yes"],
    ["Mid", "R", "yes", "yes"],
    ["Senior", "Python", "no", "no"],
    ["Senior", "R", "yes", "no"],
    ["Junior", "Python", "yes", "no"],
    ["Senior", "Python", "yes", "yes"],
    ["Mid", "Python", "no", "yes"],
    ["Mid", "Java", "yes", "no"],
    ["Junior", "Python", "no", "yes"]
]

y = ["False", "False", "True", "True", "True", "False", "True",
     "False", "True", "True", "True", "True", "True", "False"]

# N : number of decision trees
# M: most accurate trees of N trees
# F: number of the remaining attributes that are candidates to partition on


# step 1: use strat cross validation to split test and train set

# step 2:
# for N samples:
#   use bootstrapping on remainder to split sample into training and validation set
#       build tree using training set
#           - build decision tree by randomly slecting F of the remaining attributes to partition on
#       measure performance/accuracy of the tree using the validation set

# step 3:
# get M best trees based on performance scores


# predict
# use M best trees, make prediction for each instance in test set(step 1) using majority voting


# split = third of size of dataset


X_train1, X_test1, y_train1, y_test1 = myevaluation.train_test_split(
    X, y)
test_set = [X_test1[i] + [y_test1[i]] for i in range(len(X_test1))]
remainder_set = [X_train1[i] + [y_train1[i]]
                 for i in range(len(X_train1))]

random_forest = MyRandomForestClassifier(5, 2, 3)
random_forest.fit(remainder_set, test_set)
y_predicted = random_forest.predict()
print(y_predicted)
print(y_test1)