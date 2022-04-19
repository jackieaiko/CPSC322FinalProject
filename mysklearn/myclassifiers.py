"""myclassifiers.py
"""
from dis import dis
import operator
from mysklearn import myutils
from mysklearn.mysimplelinearregressor import MySimpleLinearRegressor


class MySimpleLinearRegressionClassifier:
    """Represents a simple linear regression classifier that discretizes
        predictions from a simple linear regressor (see MySimpleLinearRegressor).

    Attributes:
        discretizer(function): a function that discretizes a numeric value into
            a string label. The function's signature is func(obj) -> obj
        regressor(MySimpleLinearRegressor): the underlying regression model that
            fits a line to x and y data

    Notes:
        Terminology: instance = sample = row and attribute = feature = column
    """

    def __init__(self, discretizer, regressor=None):
        """Initializer for MySimpleLinearClassifier.

        Args:
            discretizer(function): a function that discretizes a numeric value into
                a string label. The function's signature is func(obj) -> obj
            regressor(MySimpleLinearRegressor): the underlying regression model that
                fits a line to x and y data (None if to be created in fit())
        """
        self.discretizer = discretizer
        self.regressor = regressor

    def fit(self, X_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """
        self.regressor = MySimpleLinearRegressor()
        self.regressor.fit(X_train, y_train)

    def predict(self, X_test):
        """Makes predictions for test samples in X_test by applying discretizer
            to the numeric predictions from regressor.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_nums = self.regressor.predict(X_test)
        y_predicted = self.discretizer(y_nums)
        return y_predicted


class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """

    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train
        self.y_train = y_train

    def kneighbors(self, X_test):
        """Determines the k closes neighbors of each test instance.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        row_indexes_dists = []
        for i, train_instance in enumerate(self.X_train):
            dist = myutils.compute_euclidean_distance(train_instance, X_test)
            row_indexes_dists.append([i, dist])

        row_indexes_dists.sort(key=operator.itemgetter(-1))

        k = self.n_neighbors
        top_k = row_indexes_dists[:k]

        distances = []
        neighbor_indices = []
        for row in top_k:
            neighbor_indices.append(row[0])
            distances.append(float("{:.2f}".format(row[1])))

        return distances, neighbor_indices

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        distances, neighbor_indices = self.kneighbors(X_test)

        k_closest = []
        for i in neighbor_indices:
            k_closest.append(self.y_train[i])

        values, counts = myutils.get_frequencies(k_closest)

        max_val = 0
        y_predicted = ""
        for i, _ in enumerate(counts):
            if counts[i] > max_val:
                max_val = counts[i]
                y_predicted = str(values[i])

        return [y_predicted]


class MyDummyClassifier:
    """Represents a "dummy" classifier using the "most_frequent" strategy.
        The most_frequent strategy is a Zero-R classifier, meaning it ignores
        X_train and produces zero "rules" from it. Instead, it only uses
        y_train to see what the most frequent class label is. That is
        always the dummy classifier's prediction, regardless of X_test.

    Attributes:
        most_common_label(obj): whatever the most frequent class label in the
            y_train passed into fit()

    Notes:
        Loosely based on sklearn's DummyClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
    """

    def __init__(self):
        """Initializer for DummyClassifier.

        """
        self.most_common_label = None

    def fit(self, X_train, y_train):
        """Fits a dummy classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Zero-R only predicts the most frequent class label, this method
                only saves the most frequent class label.
        """
        values, counts = myutils.get_frequencies(y_train)

        max_val = 0
        for i, _ in enumerate(counts):
            if counts[i] > max_val:
                max_val = counts[i]
                self.most_common_label = str(values[i])

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """

        y_predicted = []
        for _ in X_test:
            y_predicted.append([self.most_common_label])
        # ex. 4 yes's  if X_test =  [[] [] [] []] no matter contents
        return y_predicted


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
        
class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.
    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.
    Notes:
        Loosely based on sklearn's DecisionTreeClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.tree = None

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.
        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            On a majority vote tie, choose first attribute value based on attribute domain ordering.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        self.X_train = X_train.copy()
        train = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
        train.pop(0)
        header = X_train.pop(0)
        available_att = header.copy()
        att_domains = myutils.get_attribute_domains(header,X_train)
        self.y_train = y_train
        # next, make a copy of your header .... tdidt() is going to modify the list
        # also recall: python is pass by object reference
        tree = myutils.tdidt(train,available_att,header,att_domains,X_train)
        self.tree = tree

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        X_train = self.X_train.copy()
        header = X_train.pop(0)
        for test in X_test:
            prediction = myutils.recurse_tree(test,"",self.tree,header)
            y_predicted.append(prediction)
        return y_predicted

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format
        "IF att == val AND ... THEN class = label", one rule on each line.
        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        X_train = self.X_train.copy()
        header = X_train.pop(0)
        attribute_names = header
        myutils.print_tree(self.tree,0,"",class_name)

    # BONUS method
    def visualize_tree(self, dot_fname, pdf_fname, attribute_names=None):
        """BONUS: Visualizes a tree via the open source Graphviz graph visualization package and
        its DOT graph language (produces .dot and .pdf files).
        Args:
            dot_fname(str): The name of the .dot output file.
            pdf_fname(str): The name of the .pdf output file generated from the .dot file.
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).
        Notes:
            Graphviz: https://graphviz.org/
            DOT language: https://graphviz.org/doc/info/lang.html
            You will need to install graphviz in the Docker container as shown in class to complete this method.
        """
        with open(dot_fname, "w") as file:
            file.write("graph g {")
            myutils.bonus_graphviz(self.tree,0,[],file)
            file.write("}")
        cmd = "dot -Tpdf -o " + pdf_fname + " " + dot_fname
        os.system(cmd)