"""
Programmer: Jackie Ramsey
Class: CPSC 322-01, Spring 2022
Programming Assignment #4
3/2/22

Description: This program stores functions
"""
import csv
import math
import numpy as np

np.random.seed(4)
def compute_euclidean_distance(v1, v2):
    """computes the distance of v1 and v2 at each instance

    Args:
        v1: dataset of first attribute
        v2: dataset of second attribute

    Returns:
        dist: distances for each instance
    """
    dist = 0
    for idx_and_elem in enumerate(v1):
        i = idx_and_elem[0]
        if isinstance(v1[i], str):
            if v1[i] != v2[i]:
                dist += 1
        else:
            dist += (v1[i] - v2[i])**2
    return np.sqrt(dist)


def get_labels(list_of_val):
    """ returns unique values as labels and their freq
        Args:
            list_of_val(list of str): list of values
        Returns:
            label(list of str): list of unique values as labels
            freq(list of int): frequency of the unique val parallel to labels
    """
    label = []
    freq = []
    for val in list_of_val:
        if val in label:
            idx = label.index(val)
            freq[idx] += 1
        else:
            label.append(val)
            freq.append(1)
    return label, freq


def get_column(X, index):
    """ returns the column given the index
        Args:
            X(list of list of str): 2D table
            index(int): integer of index
        Returns:
            col(list of str): the particular column at the index of the table
    """
    col = []
    for rows in X:
        col.append(rows[index])
    return col


def get_frequencies(col):
    """finds frequencies of categorical groupings of data

    Args:
        table: dataset
        header: column names
        col_name: column name to be extracted

    Returns:
        values: groupings of column
        counts: subdatasets of groupings
    """
    col.sort()

    values = []
    counts = []
    for value in col:
        if value in values:
            counts[-1] += 1
        else:
            values.append(value)
            counts.append(1)

    return values, counts


def find_column(X_train, y_train, col_index, ci):
    """Extracts column from table

    Args:
        table: dataset
        header: column names
        col_name: column name to be extracted

    Returns:
        col: list of column values
    """
    col = []
    for i, _ in enumerate(y_train):
        if y_train[i] == ci:
            value = X_train[i][col_index]
            col.append(value)

    return col


def find1_column(X_train, col_index):
    """Extracts column from table

    Args:
        table: dataset
        header: column names
        col_name: column name to be extracted

    Returns:
        col: list of column values
    """
    col = []
    for row in X_train:
        value = row[col_index]
        if value != "NA":
            col.append(value)
    return col


# def recurse_tree(x_val, att_val, tree, header):
#     """recurse the tree
#         Args:
#             x_val (list of str/int) : the instance to recurse on/predict
#             att_val (str): attribute value
#             header (list of str): header of the table
#             tree (ND-array): tree represented in nested list
#         Returns:
#             prediction(str): class label
#         """
#     prediction = ""
#     # base case:
#     if tree[0] == "Value" and tree[1] == att_val and tree[2][0] != "Attribute":
#         for i in range(2, len(tree)):
#             if tree[i][0] == "Leaf":
#                 prediction = tree[i][1]
#                 return prediction
#     else:
#         if att_val != "":
#             tree = tree[2]
#         if tree[0] == "Attribute":
#             print("hedaer ere",header,"tree_i",tree[1])
#             index = header.index(tree[1])
#             att_val = x_val[index]
#             for i in range(2, len(tree)):
#                 if tree[i][1] == att_val:
    #                 prediction = recurse_tree(x_val, att_val, tree[i], header)
    # return prediction


def print_tree(tree, depth, rule_str, class_name):
    """print the tree
        Args:
            tree (ND-array): tree represented in nested list
            depth (int) : current depth of the tree
            rule_str(str): string of the rule
            class_name(str) : predicted class label name
    """
    if tree[0] == "Leaf":
        rule_str += "THEN " + str(class_name) + " = " + str(tree[1])
        print(rule_str)
        return rule_str
    else:
        if tree[0] == "Value":
            rule_str += " == " + str(tree[1]) + " "
            print_tree(tree[2], depth+1, rule_str, class_name)
        elif tree[0] == "Attribute":
            if depth == 0:
                rule_str = "IF " + str(tree[1])
            else:
                rule_str += " AND " + str(tree[1])
            for i in range(2, len(tree)):
                print_tree(tree[i], depth+1, rule_str, class_name)


def bonus_graphviz(tree, depth, labels, outfile, value=None, prev_att=None):
    """helper function for traversing tree for bonus question to print the tree in graphviz manner
        Args:
            tree (ND-array): tree represented in nested list
            depth (int) : current depth of the tree
            labels (list of str): list of node values, to prevent duplicates example many have True and False
            value (str): value of label (branch label)
            prev_att (str): the parent att of the current node
    """
    if tree[0] == "Leaf":
        box_name = tree[1]
        if tree[1] in labels:
            for i in range(1, 100):
                box_name = tree[1] + str(i)
                if box_name not in labels:
                    labels.append(box_name)
                    break
        else:
            labels.append(box_name)
        node_name = box_name + " [shape=box,label=" + str(tree[1]) + "]"
        outfile.write(node_name)
        branch = prev_att + " -- " + box_name + "[label=" + str(value) + "]"
        outfile.write(branch)
    else:
        if tree[1] in labels:
            for i in range(1, 100):
                box_name = tree[1] + str(i)
                if box_name not in labels:
                    labels.append(box_name)
                    break
        else:
            box_name = tree[1]
            labels.append(box_name)
        node_name = box_name + " [shape=box,label=" + str(tree[1]) + "]"
        outfile.write(node_name)
        if prev_att is not None:
            branch = prev_att + " -- " + \
                str(box_name) + "[label=" + str(value) + "]"
            outfile.write(branch)
        for i in range(2, len(tree)):
            bonus_graphviz(tree[i][2], depth, labels,
                           outfile, tree[i][1], box_name)


# def equal_trees(tree1, tree2):
#     """recursively check is both trees are equal
#         Args:
#             tree1 (ND-array): tree 1 represented in nested list
#             tree2 (ND-array): tree 2 represented in nested list
#     """
#     if tree2[0] == "Leaf":
#         if tree2[1] != tree1[1] or tree2[2] != tree1[2] or tree2[3] != tree1[3]:
#             return False
#         return True
#     else:
#         if tree2[0] == "Attribute":
#             if tree2[1] != tree1[1]:
#                 return False
#             for i in range(2, len(tree2)):
#                 if tree2[i] != tree1[i]:
#                     return False
#                 else:
#                     is_true = equal_trees(tree1[i][2], tree2[i][2])
#                     if is_true is False:
#                         return False
#     return True


def find_max(label, freqs):
    """ find max finds the label associated with the highest frequency given a list of frequencies
        Args:
            label(list of str): list of labels
            freqs(list of int): frequency of each label(parallel to labels)
        Returns:
            max_label(str): label of the highest frequency
    """
    max_f = 0
    max_label = ""
    for i, freq in enumerate(freqs):
        if freq > max_f:
            max_f = freq
            max_label = label[i]
    return max_label


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


def find_majority(index, table):
    unique_instances = []
    for row in table:
        if row[index] not in unique_instances:
            unique_instances.append(row[index])

    count_instances = [[] for _ in unique_instances]

    for row in table:
        count_instances[unique_instances.index(row[index])].append(1)

    sum_instances = []
    for row in count_instances:
        sum_instances.append(sum(row))

    majority_vote = unique_instances[sum_instances.index(max(sum_instances))]

    return majority_vote


def select_attribute(instances, attributes):
    """selects the attribute index to partition on

    Args:
        instances: available instances
        attribute: available attributes to select from

    Returns:
       attributes[rand_index]: index of attribute
    """
    select_min_entropy = []
    for i in attributes:
        attribute_types = []
        # find all attribute instance types
        for row in instances:
            if row[i] not in attribute_types:
                attribute_types.append(row[i])
        attribute_instances = [[] for _ in attribute_types]
        # find amount for each attribute
        for row in instances:
            index_att = attribute_types.index(row[i])
            attribute_instances[index_att].append(1)

        class_types = []
        for values in instances:
            if values[-1] not in class_types:
                class_types.append(values[-1])
        class_type_check = [[[] for _ in class_types] for _ in attribute_types]

        for j, _ in enumerate(instances):
            class_type_check[attribute_types.index(
                instances[j][i])][class_types.index(instances[j][-1])].append(1)

        # calculate smallest E_new
        enew = 0
        for entropy_att, _ in enumerate(class_type_check):
            entropy = 0
            for class_entropy in range(len(class_type_check[entropy_att])):
                val_instance = sum(
                    class_type_check[entropy_att][class_entropy])
                einstance = val_instance / \
                    sum(attribute_instances[entropy_att])
                if einstance != 0:
                    entropy += -1 * einstance * math.log(einstance, 2)
            enew += entropy * \
                sum(attribute_instances[entropy_att]) / len(instances)
        select_min_entropy.append(enew)

    min_index = select_min_entropy.index(min(select_min_entropy))
    return attributes[min_index]


def partition_instances(instances, split_attribute, X_train):
    """partitions list in dictionary type

    Args:
        instances: available isntances to be patitioned
        split_attribute: attribute that will be partitioned on

    Returns:
       partitions: instance partitions
    """
    # lets use a dictionary
    partitions = {}  # key (string): value (subtable)
    # att_index = header.index(split_attribute) # e.g. 0 for level
    attribute_domains = {}
    for l, _ in enumerate(X_train[0]):
        no_repeats = []
        for row in X_train:
            if str(row[l]) not in no_repeats:
                no_repeats.append(str(row[l]))
        attribute_domains[l] = no_repeats

    att_index = split_attribute
    # e.g. ["Junior", "Mid", "Senior"]
    att_domain = attribute_domains[att_index]
    for att_value in att_domain:
        partitions[att_value] = []
        # task: finish
        for instance in instances:
            if instance[att_index] == att_value:
                partitions[att_value].append(instance)

    return partitions


def all_same_class(instances):
    """checks if all instances have the same label

    Args:
        instances: instances
        class_index: label value being checked

    Returns:
       True or False
    """
    check_same = instances[0][-1]
    for attribute_vals in instances:
        if attribute_vals[-1] != check_same:
            return False
    return True


def second_case(att_partition, current_instances, value_subtree, tree):
    """does majority vote for leaf

    Args:
        att_partition: instances to be partitioned
        current_instances: available attributes to partition
        value_subtree: subtree
        tree: tree
    """
    classifiers = []
    for value_class in att_partition:
        if value_class[-1] not in classifiers:
            classifiers.append(value_class[-1])
    # find amount for each classifier

    find_majority = [[] for _ in classifiers]
    for value_class in att_partition:
        find_majority[classifiers.index(value_class[-1])].append(1)

    # find max amount
    max_val = 0
    for count in find_majority:
        total_sum = sum(count)
        if total_sum > max_val:
            majority_rule = classifiers[find_majority.index(count)]

    leaf_node = ["Leaf", majority_rule, len(
        att_partition), len(current_instances)]
    value_subtree.append(leaf_node)
    tree.append(value_subtree)


def tdidt(current_instances, available_attributes, X_train):
    """recursively builds decision tree

    Args:
        current_instances: instances to be partitioned
        available_attributes: available attributes to partition

    Returns:
       tree: the updated tree
    """
    # select an attribute to split on
    attribute = select_attribute(current_instances, available_attributes)
    available_attributes.remove(attribute)
    tree = ["Attribute", "att" + str(attribute)]
    # group data by attribute domains (creates pairwise disjoint partitions)
    partitions = partition_instances(current_instances, attribute, X_train)
    # for each partition, repeat unless one of the following occurs (base case)
    for att_value, att_partition in partitions.items():
        value_subtree = ["Value", att_value]

        #    CASE 1: all class labels of the partition are the same => make a leaf node
        if len(att_partition) > 0 and all_same_class(att_partition):
            leaf_node = ["Leaf", att_partition[0][-1],
                         len(att_partition), len(current_instances)]
            value_subtree.append(leaf_node)
            tree.append(value_subtree)

        #    CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
        elif len(att_partition) > 0 and len(available_attributes) == 0:
            second_case(att_partition, current_instances, value_subtree, tree)

        #    CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
        elif len(att_partition) == 0:
            return None

        else:  # the previous conditions are all false... recurse!!
            subtree = tdidt(
                att_partition, available_attributes.copy(), X_train)
            if subtree is None:
                second_case(att_partition, current_instances,
                            value_subtree, tree)
            else:
                value_subtree.append(subtree)
                tree.append(value_subtree)
    return tree


def tdidt_predict(header, tree, instance):
    """predicts instances using decisions tree

    Args:
        header: attribute labels
        tree: tree after fit() called
        instance: X_test instance

    Returns:
       tree[1]: classification at leaf
    """
    # recursively traverse tree to make a prediction
    # are we at a leaf node (base case) or attribute node?
    info_type = tree[0]
    if info_type == "Leaf":
        return tree[1]  # label
    # we are at an attribute
    # find attribute value match for instance
    # for loop
    att_index = header.index(tree[1])
    for i in range(2, len(tree)):
        value_list = tree[i]
        if value_list[1] == instance[att_index]:
            return tdidt_predict(header, value_list[2], instance)
