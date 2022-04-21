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
                dist +=1
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
            freq[idx] +=1
        else:
            label.append(val)
            freq.append(1)
    return label, freq

def get_column(X,index):
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

def select_attribute(instances, attributes,header,X_train):
    """chooses the attribute to split on based on entropy value
        Args:
            instances (list of list of str/int) : represent the instances in this split
            attributes (list of str): attributes to split on
            header (list of str): header of the table
            X_train (list of list of str/int): represent the orignal X dataset
        Returns:
            split_att(str): attribute to split on
        Notes:
            entropy value calculated
        """
    split_att = attributes[0]
    split_att_ent = 500
    for att in attributes:
        index = header.index(att)
        labels = []
        labels_indexes = []
        total_ent = 0
        for i,instance in enumerate(instances):
            if instance[index] not in labels:
                labels.append(instance[index])
                labels_indexes.append([i])
            else:
                i_labels = labels.index(instance[index])
                labels_indexes[i_labels].append(i)
        for label_indexes in labels_indexes:
            class_labels = []
            class_freq = []
            for elem_index in label_indexes:
                if instances[elem_index][-1] not in class_labels:
                    class_labels.append(instances[elem_index][-1])
                    class_freq.append(1)
                else:
                    idx = class_labels.index(instances[elem_index][-1])
                    class_freq[idx] +=1
            sums = 0
            for label_idx,class_label in enumerate(class_labels):
                val = class_freq[label_idx]/len(label_indexes)
                sums += (-val) * (math.log2(val))
            total_ent += (len(label_indexes) / len(X_train)) * sums
        if total_ent < split_att_ent:
            split_att_ent = total_ent
            split_att = att
    return split_att

def partition_instances(instances, split_atribute,header,attribute_domains):
    """group the instances together
        Args:
            instances (list of list of str/int) : represent the instances in this split
            split_attribute(str): attribute to group instances
            header (list of str): header of the table
            attribute_domains (list of str): attribute domains of that attrubute
        partitions (dictionary): attribute domain as key and instances as values
     """
    # lets use a dictionary
    partitions = {} # key string mapping to subvalue pairs
    att_index = header.index(split_atribute) # e.g: 0 for level
    att_domain = attribute_domains[split_atribute] # e.g: ["Junior","Mid","Senior"]
    for att_val in att_domain:
        partitions[att_val] = []
        for instance in instances:
            if instance[att_index] == att_val:
                partitions[att_val].append(instance)

    return partitions

def majority_node(instances):
    """the class value that most instances have
        Args:
            instances (list of list of str/int) : represent the instances in this split
        Returns:
            label(str): most occuring/frequent class label of the instances
    """
    labels = []
    freqs = []
    for instance in instances:
        if instance[-1] not in labels:
            labels.append(instance[-1])
            freqs.append(1)
        else:
            i = labels.index(instance[-1])
            freqs[i] +=1
    min = 100
    min_index = -1
    for i,freq in enumerate(freqs):
        if freq < min:
            min = freq
            min_index = i
        elif freq == min:
            if labels[min_index] > labels[i]:
                min_index = i
    return labels[min_index]

def get_attribute_domains(header,table):
    """returns the attribute domains in each attribute in alphabetical order
        Args:
            table (list of list of str/int) : represent the whole dataset table
            header (list of str): header of the table
        Returns:
            dict_domains(dictionary): list of attributes as keys and and its domains as values
    """
    dict_domains = {}
    arr_vals = []
    for i,elem in enumerate(header):
        arr_vals = []
        for row in table:
            if row[i] not in arr_vals:
                arr_vals.append(row[i])
        arr_vals.sort()
        dict_domains.update({elem:arr_vals})

    return dict_domains

def tdidt(current_instances, available_attributes,header,attribute_domains,X_train):
    """responsible for fitting the dataset into a tree using the tdidt algorithmn, will decide how to split into a tree
        Args:
            current_instances (list of list of str/int) : represent the instances in this split
            available_attributes (list of str): attributes that can still be split on
            header (list of str): header of the table
            attribute_domains (dictionary): attribute domains for all the attributes
            X_train (list of list of str/int): represent the orignal X dataset
        Returns:
            tree(N-D array of list or int): tree represented in nested lists
    """
    # basic approach (uses recursion!!):
    # select an attribute to split on
    attribute = select_attribute(current_instances,available_attributes,header,X_train)
    #print("split on:",attribute)
    # remove split att
    available_attributes.remove(attribute)
    tree = ["Attribute", attribute]
    # group data by attribute domains (creates pairwise disjoint partitions)
    partition = partition_instances(current_instances,attribute,header,attribute_domains)
    # for each partition, repeat unless one of the following occurs (base case)
    for att_val, att_partition in partition.items():
        value_subtree = ["Value", att_val]
    #   CASE 1: all class labels of the partition are the same => make a leaf node
        if len(att_partition) > 0 and all_same_class(att_partition):
            #print("CASE 1, ALL SAME CATEGORIES")
            node = ["Leaf",att_partition[0][-1],len(att_partition), len(current_instances)]
            value_subtree.append(node)
            tree.append(value_subtree)
        #    CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
        elif len(att_partition) > 0 and len(available_attributes) == 0:
            #print("CASE 2, NO MORE ATT")
            node_val = majority_node(att_partition)
            node = ["Leaf",node_val,len(att_partition),len(current_instances)]
            value_subtree.append(node)
            tree.append(value_subtree)
        #    CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
        elif len(att_partition) == 0:
            #print("CASE 3 empty partition")
            return None
        else:
            subtree = tdidt(att_partition,available_attributes.copy(),header,attribute_domains,X_train)
            if subtree is None:
                val = 0
                node_val = majority_node(att_partition)
                for rows in att_partition:
                    if rows[-1] == node_val:
                        val+=1
                subtree = ["Leaf",node_val,len(att_partition), len(current_instances)]
            value_subtree.append(subtree)
            tree.append(value_subtree)
    return tree

def all_same_class(att_partition):
    """checks if the istances have same class label
        Args:
            att_partition (list of list of str/int) : represent the instances in this split
        Returns:
            boolean of True or False, True means all same, false otherwise
d
        """
    label = None
    for partition in att_partition:
        if label is None:
            label = partition[-1]
        else:
            if label != partition[-1]:
                return False
    return True

def recurse_tree(x_val,att_val,tree,header):
    """recurse the tree
        Args:
            x_val (list of str/int) : the instance to recurse on/predict
            att_val (str): attribute value
            header (list of str): header of the table
            tree (ND-array): tree represented in nested list
        Returns:
            prediction(str): class label
        """
    prediction = ""
    #base case:
    if tree[0] == "Value" and tree[1] == att_val and tree[2][0]!="Attribute":
        for i in range(2,len(tree)):
            if tree[i][0] == "Leaf":
                prediction = tree[i][1]  
                return prediction
    else:
        if att_val != "":
            tree = tree[2]
        if tree[0] == "Attribute":
            index = header.index(tree[1])
            att_val = x_val[index]
            for i in range(2,len(tree)):
                if tree[i][1] == att_val:
                    prediction = recurse_tree(x_val,att_val,tree[i],header)
    return prediction
            
def print_tree(tree,depth,rule_str,class_name):
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
            print_tree(tree[2],depth+1,rule_str,class_name)
        elif tree[0] == "Attribute":
            if depth == 0:
                rule_str = "IF " + str(tree[1])
            else:
                rule_str += " AND " + str(tree[1])
            for i in range(2,len(tree)):
                print_tree(tree[i],depth+1,rule_str,class_name)

def bonus_graphviz(tree,depth,labels,outfile,value=None,prev_att=None):
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
            for i in range(1,100):
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
            for i in range(1,100):
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
            branch = prev_att + " -- " + str(box_name) + "[label=" + str(value) + "]"
            outfile.write(branch)
        for i in range(2,len(tree)):
            bonus_graphviz(tree[i][2],depth,labels,outfile,tree[i][1],box_name)

def equal_trees(tree1,tree2):
    """recursively check is both trees are equal
        Args:
            tree1 (ND-array): tree 1 represented in nested list
            tree2 (ND-array): tree 2 represented in nested list
    """
    if tree2[0] == "Leaf":
        if tree2[1] != tree1[1] or tree2[2] != tree1[2] or tree2[3] != tree1[3]:
            return False
        return True
    else:
        if tree2[0] == "Attribute":
            if tree2[1] != tree1[1]:
                return False
            for i in range(2,len(tree2)):
                if tree2[i] != tree1[i]:
                    return False
                else:
                    is_true = equal_trees(tree1[i][2],tree2[i][2])
                    if is_true is False:
                        return False
    return True

def find_max(label,freqs):
    """ find max finds the label associated with the highest frequency given a list of frequencies
        Args:
            label(list of str): list of labels
            freqs(list of int): frequency of each label(parallel to labels)
        Returns:
            max_label(str): label of the highest frequency
    """
    max_f = 0
    max_label = ""
    for i,freq in enumerate(freqs):
        if freq > max_f:
            max_f = freq
            max_label = label[i]
    return max_label