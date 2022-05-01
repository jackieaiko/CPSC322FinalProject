'''
testing MyRandomForestClassifier
'''
import numpy as np
from sympy import N
import mysklearn.myevaluation as myevaluation
from mysklearn.myclassifiers import MySimpleLinearRegressionClassifier,\
    MyKNeighborsClassifier,\
    MyDummyClassifier,\
    MyNaiveBayesClassifier, \
    MyDecisionTreeClassifier, \
    MyRandomForestClassifier

X_train_class_example1 = [[1, 1], [1, 0], [0.33, 0], [0, 0]]
y_train_class_example1 = ["bad", "bad", "good", "good"]

X_train_class_example2 = [
    [3, 2],
    [6, 6],
    [4, 1],
    [4, 4],
    [1, 2],
    [2, 0],
    [0, 3],
    [1, 6]]
y_train_class_example2 = ["no", "yes",
                          "no", "no", "yes", "no", "yes", "yes"]


def test_kneighbors_classifier_kneighbors():
    """tests kneighbors function from MyKNeighborsClassifier
    """
    function_kneighbor = MyKNeighborsClassifier()

    X_test = [[0.33, 1]]
    test_distances = [0.67, 1.00, 1.05]
    test_neighbor_indices = [0, 2, 3]

    function_kneighbor.fit(X_train_class_example1, y_train_class_example1)
    func_dist, funct_indices = function_kneighbor.kneighbors(X_test)

    for i,test_distance in enumerate(test_distances):
        assert np.isclose(round(func_dist[0][i],2), test_distance)
        assert np.isclose(round(funct_indices[0][i],2), test_neighbor_indices[i])

    X_test = [[2, 3]]
    test_distances = [1.41, 1.41, 2.00]
    test_neighbor_indices = [0, 4, 6]

    function_kneighbor.fit(X_train_class_example2, y_train_class_example2)
    func_dist, funct_indices = function_kneighbor.kneighbors(X_test)

    for i,test_distance in enumerate(test_distances):
        assert np.isclose(round(func_dist[0][i],2), test_distance)
        assert np.isclose(round(funct_indices[0][i],2), test_neighbor_indices[i])


def test_kneighbors_classifier_predict():
    #     """tests predict function from MyKNeighborsClassifier
    #     """
    function_kpredict = MyKNeighborsClassifier()

    X_test = [[0.33, 1]]
    test_y_predicted = ["good"]

    function_kpredict.fit(X_train_class_example1, y_train_class_example1)
    y_predicted = function_kpredict.predict(X_test)
    for y in y_predicted:
        assert y == test_y_predicted

    X_test = [[2, 3]]
    test_y_predicted = ["yes"]

    function_kpredict.fit(X_train_class_example2, y_train_class_example2)
    y_predicted = function_kpredict.predict(X_test)
    for y in y_predicted:
        assert y == test_y_predicted


def test_dummy_classifier_fit():
    """tests fit function from MyDummyClassifier
    """
    dummy_class_fit = MyDummyClassifier()
    X_train = list(range(100))

    y_train = list(np.random.choice(
        ["yes", "no"], 100, replace=True, p=[0.7, 0.3]))
    test_most_common_label = "yes"
    dummy_class_fit.fit(X_train, y_train)
    assert test_most_common_label == dummy_class_fit.most_common_label

    y_train = list(np.random.choice(
        ["yes", "no", "maybe"], 100, replace=True, p=[0.2, 0.6, 0.2]))
    test_most_common_label = "no"
    dummy_class_fit.fit(X_train, y_train)
    assert test_most_common_label == dummy_class_fit.most_common_label

    y_train = list(np.random.choice(
        ["go", "slow", "stop"], 100, replace=True, p=[0.3, 0.6, 0.1]))
    test_most_common_label = "slow"
    dummy_class_fit.fit(X_train, y_train)
    assert test_most_common_label == dummy_class_fit.most_common_label


def test_dummy_classifier_predict():
    """tests predict function from MyDummyClassifier
    """
    dummy_class_predict = MyDummyClassifier()
    X_train = list(range(100))
    X_test = [1, 2, 3, 4]

    y_train = list(np.random.choice(
        ["yes", "no"], 100, replace=True, p=[0.7, 0.3]))
    dummy_class_predict.fit(X_train, y_train)
    test_dummy_predict = [["yes"], ["yes"], ["yes"], ["yes"]]
    for i, _ in enumerate(test_dummy_predict):
        assert dummy_class_predict.predict(X_test)[i] == test_dummy_predict[i]

    y_train = list(np.random.choice(
        ["yes", "no", "maybe"], 100, replace=True, p=[0.2, 0.6, 0.2]))
    dummy_class_predict.fit(X_train, y_train)
    test_dummy_predict = [["no"], ["no"], ["no"], ["no"]]
    for i, _ in enumerate(test_dummy_predict):
        assert dummy_class_predict.predict(X_test)[i] == test_dummy_predict[i]

    y_train = list(np.random.choice(
        ["go", "slow", "stop"], 100, replace=True, p=[0.3, 0.6, 0.1]))
    dummy_class_predict.fit(X_train, y_train)
    test_dummy_predict = [["slow"], ["slow"], ["slow"], ["slow"]]
    for i, _ in enumerate(test_dummy_predict):
        assert dummy_class_predict.predict(X_test)[i] == test_dummy_predict[i]


def discretized_values(X_test):
    """converts numerical data into classifications

    Args:
        X_test: testing data

    Returns:
        discretized_train: classified X_test
    """
    discretized_train = []
    for i, _ in enumerate(X_test):
        if X_test[i] >= 100:
            discretized_train.append("high")
        else:
            discretized_train.append("low")
    return discretized_train


# in-class Naive Bayes example (lab task #1)
X_train_inclass_example = [
    [1, 5],  # yes
    [2, 6],  # yes
    [1, 5],  # no
    [1, 5],  # no
    [1, 6],  # yes
    [2, 6],  # no
    [1, 5],  # yes
    [1, 6]  # yes
]
y_train_inclass_example = ["yes", "yes", "no", "no", "yes", "no", "yes", "yes"]

# RQ5 (fake) iPhone purchases dataset
X_train_iphone = [
    [1, 3, "fair"],
    [1, 3, "excellent"],
    [2, 3, "fair"],
    [2, 2, "fair"],
    [2, 1, "fair"],
    [2, 1, "excellent"],
    [2, 1, "excellent"],
    [1, 2, "fair"],
    [1, 1, "fair"],
    [2, 2, "fair"],
    [1, 2, "excellent"],
    [2, 2, "excellent"],
    [2, 3, "fair"],
    [2, 2, "excellent"],
    [2, 3, "fair"]
]
y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no",
                  "yes", "yes", "yes", "yes", "yes", "no", "yes"]


def test_naive_bayes_classifier_fit():
    """tests fit function from MyNaiveBayesClassifier
    """
    expected_priors = {'yes': 0.625, 'no': 0.375}
    expected_posteriors = [{1: {'yes': 0.8, 'no': 0.667}, 2: {'yes': 0.2, 'no': 0.333}},
                           {5: {'yes': 0.4, 'no': 0.667}, 6: {'yes': 0.6, 'no': 0.333}}]

    naive_bayes_fit = MyNaiveBayesClassifier()
    naive_bayes_fit.fit(X_train_inclass_example, y_train_inclass_example)
    print(naive_bayes_fit.priors)
    print(naive_bayes_fit.posteriors)
    assert expected_priors == naive_bayes_fit.priors
    assert expected_posteriors == naive_bayes_fit.posteriors

    expected_priors = {'no': 0.333, 'yes': 0.667}
    expected_posteriors = [{1: {'no': 0.6, 'yes': 0.2}, 2: {'no': 0.4, 'yes': 0.8}},
                           {3: {'no': 0.4, 'yes': 0.3}, 1: {'no': 0.2,
                                                            'yes': 0.3}, 2: {'no': 0.4, 'yes': 0.4}},
                           {'fair': {'no': 0.4, 'yes': 0.7}, 'excellent': {'no': 0.6, 'yes': 0.3}}]

    naive_bayes_fit = MyNaiveBayesClassifier()
    naive_bayes_fit.fit(X_train_iphone, y_train_iphone)
    assert expected_priors == naive_bayes_fit.priors
    assert expected_posteriors == naive_bayes_fit.posteriors


def test_naive_bayes_classifier_predict():
    """tests predict function from MyNaiveBayesClassifier
    """
    naive_bayes_fit = MyNaiveBayesClassifier()
    naive_bayes_fit.fit(X_train_inclass_example, y_train_inclass_example)
    X_test = [[1, 5]]
    expected_predictions = [["yes"]]
    assert naive_bayes_fit.predict(X_test) == expected_predictions

    naive_bayes_fit = MyNaiveBayesClassifier()
    naive_bayes_fit.fit(X_train_iphone, y_train_iphone)
    X_test = [[2, 2, "fair"], [1, 1, "excellent"]]
    expected_predictions = [['yes'], ['no']]
    assert naive_bayes_fit.predict(X_test) == expected_predictions


# interview dataset
X_train_interview = [
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
y_train_interview = ["False", "False", "True", "True", "True", "False", "True",
                     "False", "True", "True", "True", "True", "True", "False"]


def test_decision_tree_classifier_fit():
    decision_tree = MyDecisionTreeClassifier()
    decision_tree.fit(X_train_interview, y_train_interview,
                      len(y_train_interview))

    tree_job = \
        ['Attribute', 'att0',
         ['Value', 'Senior',
          ['Attribute', 'att2',
           ['Value', 'no',
            ['Leaf', 'False', 3, 5]],
              ['Value', 'yes',
               ['Leaf', 'True', 2, 5]]]],
            ['Value', 'Mid',
             ['Leaf', 'True', 4, 14]],
            ['Value', 'Junior',
             ['Attribute', 'att3',
              ['Value', 'no',
               ['Leaf', 'True', 3, 5]],
                 ['Value', 'yes',
                  ['Leaf', 'False', 2, 5]]]]]
    assert tree_job == decision_tree.tree


def test_decision_tree_classifier_predict():
    decision_tree = MyDecisionTreeClassifier()
    decision_tree.fit(X_train_interview, y_train_interview,
                      len(y_train_interview))
    X_test = [["Junior", "Java", "yes", "no"],
              ["Junior", "Java", "yes", "yes"]]
    expected_predict = ["True", "False"]
    predicted = decision_tree.predict(X_test)
    assert expected_predict == predicted


def test_random_forest_fit():
    
    random_forest = MyRandomForestClassifier(5, 2, 3)
    random_forest.fit(X_train_interview,y_train_interview)

    assert 5 == random_forest.n
    assert 2 == random_forest.m
    assert 3 == random_forest.f


def test_random_forest_predict():
    random_forest = MyRandomForestClassifier(5, 2, 3)
    random_forest.fit(X_train_interview,y_train_interview)
    X_test = [["Junior", "Java", "yes", "no"],
              ["Junior", "Java", "yes", "yes"]]

    y_predicted = random_forest.predict(X_test)
    assert 2 == len(y_predicted)
