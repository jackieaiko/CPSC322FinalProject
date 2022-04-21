"""
class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        priors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The prior probabilities computed for each
            label in the training set.
        posteriors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The posterior probabilities computed for each
            attribute value/label pair in the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        You may add additional instance attributes if you would like, just be sure to update this docstring
        Terminology: instance = sample = row and attribute = feature = column
    """

    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.
        """
        self.priors = None
        self.posteriors = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """
        y_train_priors = y_train.copy()
        # find priors
        self.priors = {}
        c_i, priors = myutils.get_frequencies(y_train_priors)
        for i in c_i:
            value = priors[c_i.index(i)] / len(y_train_priors)
            self.priors[i] = float("{:.2f}".format(value))

        # find frequency values of each
        c_i_cat_key = [[] for _ in c_i]
        c_i_cat_value = []
        for j in c_i:
            for i in range(len(X_train[0])):
                cur_col = myutils.find_column(X_train, y_train, i, j)
                compare_col = myutils.find1_column(X_train, i)

                a_h, posteriors = myutils.get_frequencies(cur_col)
                # for when frequency is 0
                for l in compare_col:
                    if l not in a_h:
                        a_h.append(l)
                        posteriors.append(0)

                c_i_cat_key[c_i.index(j)].append(posteriors)
                if c_i.index(j) == 0:
                    c_i_cat_value.append(a_h)

        # create dictionary
        att_names = list(range(len(X_train[0])))
        new_dict = {}
        for i in c_i:
            new_dict[i] = {}
            for j in att_names:
                new_dict[i][j] = {}

        # fill dictionary
        for i, _ in enumerate(att_names):
            for j in c_i:
                for k in c_i_cat_value[i]:
                    value = c_i_cat_key[c_i.index(
                        j)][i][c_i_cat_value[i].index(k)] / priors[c_i.index(j)]
                    new_dict[j][att_names[i]][k] = float(
                        "{:.2f}".format(value))

        self.posteriors = new_dict

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """

        decisions = []
        label_tracker = []
        for i, _ in enumerate(X_test):
            decision = []
            for label in self.posteriors:
                label_tracker.append(label)
                label_prediction = self.priors[label]
                for class_type in self.posteriors[label]:
                    label_prediction *= self.posteriors[label][class_type][X_test[i][class_type]]
                decision.append(float("{:.4f}".format(label_prediction)))
            decisions.append(decision)

        y_predicted = []
        for i in decisions:
            loc = i.index(max(i))
            y_predicted.append(label_tracker[loc])

        return y_predicted
"""
