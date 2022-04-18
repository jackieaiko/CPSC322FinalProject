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
    if type(v1[0]) == str:  # is a string
        dist = np.sqrt(sum(1 if v1[i] == v2[i] else 0 for i in range(len(v1))))
    else:
        dist = np.sqrt(sum([(v1[i] - v2[i]) ** 2 for i in range(len(v1))]))

    return dist


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
