"""myevaluation.py
"""
#from mysklearn import myutils
import random
import numpy as np


def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets based on a test set size.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
            Use random_state to seed your random number generator
                you can use the math module or use numpy for your generator
                choose one and consistently use that generator throughout your code
        shuffle(bool): whether or not to randomize the order of the instances before splitting
            Shuffle the rows in X and y before splitting and be sure to maintain the parallel order of X and y!!

    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)

    Note:
        Loosely based on sklearn's train_test_split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    random.seed(random_state)

    if shuffle:
        combined_list = list(zip(X, y))
        for i in range(len(combined_list)-1):
            index = range(len(combined_list))
            temp1, temp2 = random.sample(index, 2)
            combined_list[temp1], combined_list[temp2] = combined_list[temp2], combined_list[temp1]
        X, y = zip(*combined_list)

    if test_size < 1:
        test_size = int(np.ceil(test_size * len(y)))
    test_instance_indices = [*range(len(X)-test_size, len(X))]

    X_train = []
    for i, _ in enumerate(X):
        if i not in test_instance_indices:
            X_train.append(X[i])

    X_test = []
    for i in test_instance_indices:
        X_test.append(X[i])

    y_train = []
    for i, _ in enumerate(y):
        if i not in test_instance_indices:
            y_train.append(y[i])

    y_test = []
    for i in test_instance_indices:
        y_test.append(y[i])

    return X_train, X_test, y_train, y_test


def kfold_cross_validation(X, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds
    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold
        X_test_folds(list of list of int): The list of testing set indices for each fold

    Notes:
        The first n_samples % n_splits folds have size n_samples // n_splits + 1,
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    np.random.seed(random_state)

    indices = list(range(len(X)))
    if shuffle:
        combined_list = list(zip(X, indices))
        np.random.shuffle(combined_list)
        X, indices = zip(*combined_list)

    # empty list of test folds
    X_test_folds = []
    for i in range(n_splits):
        fold = []
        X_test_folds.append(fold)

    # max amt per fold
    amt_per_fold = int(np.ceil(len(X) / n_splits))

    # fills folds
    j = 0
    count_reset = 1
    for i in indices:
        X_test_folds[j].append(i)
        if count_reset == amt_per_fold:
            j += 1
            count_reset = 1
        else:
            count_reset += 1

    X_train_folds = []
    sub_train = []
    for i in X_test_folds:
        for j in X_test_folds:
            if i != j:
                for k in j:
                    sub_train.append(k)
        X_train_folds.append(sub_train)
        sub_train = []

    return X_train_folds, X_test_folds


def stratified_kfold_cross_validation(X, y, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into stratified cross validation folds.

    Args:
        X(list of list of obj): The list of instances (samples).
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X).
            The shape of y is n_samples
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds
    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold.
        X_test_folds(list of list of int): The list of testing set indices for each fold.

    Notes:
        Loosely based on sklearn's StratifiedKFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """
    np.random.seed(random_state)

    indices = list(range(len(X)))
    if shuffle:
        combined_list = list(zip(X, y, indices))
        np.random.shuffle(combined_list)
        X, y, indices = zip(*combined_list)

    # group by class
    groups = []
    for i in y:
        if i not in groups:
            groups.append(i)

    grouped_index = [[] for _ in groups]
    for i, _ in enumerate(y):
        val_index = groups.index(y[i])
        grouped_index[val_index].append(indices[i])

    # empty list
    X_test_folds = [[] for i in range(n_splits)]

    fold_location = 0
    for i in grouped_index:
        for j in i:
            X_test_folds[fold_location].append(j)
            if fold_location == n_splits - 1:
                fold_location = 0
            else:
                fold_location += 1

    X_train_folds = []
    xtrain = []
    for i in X_test_folds:
        for j in X_test_folds:
            if i != j:
                for k in j:
                    xtrain.append(k)
        X_train_folds.append(xtrain)
        xtrain = []

    return X_train_folds, X_test_folds


def bootstrap_sample(X, y=None, n_samples=None, random_state=None):
    """Split dataset into bootstrapped training set and out of bag test set.

    Args:
        X(list of list of obj): The list of samples
        y(list of obj): The target y values (parallel to X)
            Default is None (in this case, the calling code only wants to sample X)
        n_samples(int): Number of samples to generate. If left to None (default) this is automatically
            set to the first dimension of X.
        random_state(int): integer used for seeding a random number generator for reproducible results
    Returns:
        X_sample(list of list of obj): The list of samples
        X_out_of_bag(list of list of obj): The list of "out of bag" samples (e.g. left-over samples)
        y_sample(list of obj): The list of target y values sampled (parallel to X_sample)
            None if y is None
        y_out_of_bag(list of obj): The list of target y values "out of bag" (parallel to X_out_of_bag)
            None if y is None
    Notes:
        Loosely based on sklearn's resample():
            https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html
    """
    np.random.seed(random_state)

    if n_samples is None:
        n_samples = len(X)

    test_instance_indices = np.random.choice(
        range(len(X)), size=n_samples, replace=True)

    X_sample = []
    X_out_of_bag = []
    y_sample = []
    y_out_of_bag = []

    if y is None:
        y_sample = None
        y_out_of_bag = None
        for i in test_instance_indices:
            X_sample.append(X[i])

        for i, _ in enumerate(X):
            if i not in test_instance_indices:
                X_out_of_bag.append(X[i])
    else:
        for i in test_instance_indices:
            X_sample.append(X[i])
            y_sample.append(y[i])

        for i, _ in enumerate(X):
            if i not in test_instance_indices:
                X_out_of_bag.append(X[i])
                y_out_of_bag.append(y[i])

    return X_sample, X_out_of_bag, y_sample, y_out_of_bag


def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.
    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix
    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry
            indicates the number of samples with true label being i-th class
            and predicted label being j-th class
    Notes:
        Loosely based on sklearn's confusion_matrix():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    matrix = []
    for i, label in enumerate(labels):
        list_val = [0 for l in range(0, len(labels))]
        for j, y in enumerate(y_true):
            if y == label:
                if y_pred[j] == y:
                    list_val[i] += 1
                else:
                    idx = labels.index(y_pred[j])
                    list_val[idx] += 1
        matrix.append(list_val)
    return matrix


def accuracy_score(y_true, y_pred, normalize=True):
    """Compute the classification prediction accuracy score.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        normalize(bool): If False, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.

    Returns:
        score(float): If normalize == True, return the fraction of correctly classified samples (float),
            else returns the number of correctly classified samples (int).

    Notes:
        Loosely based on sklearn's accuracy_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    """
    count = 0
    for i, _ in enumerate(y_pred):
        if y_pred[i] == y_true[i]:
            count += 1

    if normalize:
        score = count / len(y_pred)
    else:
        score = float(count)

    return score


def binary_precision_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the precision (for binary classification). The precision is the ratio tp / (tp + fp)
        where tp is the number of true positives and fp the number of false positives.
        The precision is intuitively the ability of the classifier not to label as
        positive a sample that is negative. The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        precision(float): Precision of the positive class

    Notes:
        Loosely based on sklearn's precision_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    """
    if labels is None:
        labels = []
        for i in y_true:
            if i not in labels:
                labels.append(i)

    if pos_label is None:
        pos_label = labels[0]

    true_pos = 0
    false_pos = 0
    for i, _ in enumerate(y_true):
        if y_pred[i] == pos_label:
            if y_true[i] == pos_label:
                true_pos += 1
            elif y_true[i] != pos_label:
                false_pos += 1

    if true_pos+false_pos != 0:
        precision = true_pos / (true_pos+false_pos)
    else:
        precision = 0

    return precision


def binary_recall_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the recall (for binary classification). The recall is the ratio tp / (tp + fn) where tp is
        the number of true positives and fn the number of false negatives.
        The recall is intuitively the ability of the classifier to find all the positive samples.
        The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        recall(float): Recall of the positive class

    Notes:
        Loosely based on sklearn's recall_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    """
    if labels is None:
        labels = []
        for i in y_true:
            if i not in labels:
                labels.append(i)

    if pos_label is None:
        pos_label = labels[0]

    true_pos = 0
    false_neg = 0
    for i, _ in enumerate(y_true):
        if y_pred[i] == pos_label:
            if y_true[i] == pos_label:
                true_pos += 1
        elif y_pred[i] != pos_label:
            if y_true[i] == pos_label:
                false_neg += 1

    if true_pos+false_neg != 0:
        recall = true_pos / (true_pos+false_neg)
    else:
        recall = 0

    return recall


def binary_f1_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the F1 score (for binary classification), also known as balanced F-score or F-measure.
        The F1 score can be interpreted as a harmonic mean of the precision and recall,
        where an F1 score reaches its best value at 1 and worst score at 0.
        The relative contribution of precision and recall to the F1 score are equal.
        The formula for the F1 score is: F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        f1(float): F1 score of the positive class

    Notes:
        Loosely based on sklearn's f1_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    """
    precision = binary_precision_score(y_true, y_pred, labels, pos_label)
    recall = binary_recall_score(y_true, y_pred, labels, pos_label)

    if precision+recall != 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
    return f1
