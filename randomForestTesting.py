
from mysklearn.myclassifiers import MyKNeighborsClassifier, MyDummyClassifier, MyNaiveBayesClassifier, MyDecisionTreeClassifier
import mysklearn.myevaluation as myevaluation
from mysklearn import myutils
import numpy as np


def compute_bootstrapped_sample(table):
    n = len(table)
    train_set = []
    for _ in range(n):
        # Return random integers from low (inclusive) to high (exclusive)
        rand_index = np.random.randint(0, n)
        train_set.append(table[rand_index])

    validation_set = []
    for i in range(n):
        if table[i] not in train_set:
            validation_set.append(table[i])

    X_train = [row[0:-1] for row in train_set]
    y_train = [row[-1] for row in train_set]
    X_test = [row[0:-1] for row in validation_set]
    y_test = [[row[-1]] for row in validation_set]

    return X_train, y_train, X_test, y_test


def compute_random_subset(header, f):
    # there is a function np.random.choice()
    values_copy = header[:]  # shallow copy
    np.random.shuffle(values_copy)  # in place shuffle
    return values_copy[:f]


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

X_train, y_train, X_test, y_test = myevaluation.train_test_split(X, y)
print(y_train)
print(y_test)
test_set = [X_test[i] + [y_test[i]] for i in range(len(X_test))]
remainder = [X_train[i] + [y_train[i]] for i in range(len(X_train))]

n = 4
f = 3
m = 2

n_forest = []
n_performance = []
for i in range(n):
    X_train, y_train, X_test, y_test = compute_bootstrapped_sample(remainder)

    decision_tree_classifier = MyDecisionTreeClassifier()
    decision_tree_classifier.fit(X_train, y_train, f)

    y_predicted = decision_tree_classifier.predict(X_test)
    accuracy_score = myevaluation.accuracy_score(y_test, y_predicted)

    n_forest.append(decision_tree_classifier.tree)
    n_performance.append(accuracy_score)

# find largest values
largest_indices = sorted(range(len(n_performance)),
                         key=lambda i: n_performance[i])[-m:]

m_forest = [n_forest[i] for i in largest_indices]
print(m_forest)

# predicted_forests = []
# # for i in range(m):
# tree = m_forest[1]

# y_predicted = []
# remainder = remainder.copy()
# header = remainder.pop(0)
# for test in X_test:
#     prediction = myutils.recurse_tree(test, "", tree, header)
#     y_predicted.append([prediction])
