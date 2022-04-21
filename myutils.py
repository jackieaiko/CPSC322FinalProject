from calendar import c
from mysklearn.myclassifiers import MyDecisionTreeClassifier, MyDummyClassifier, MyKNeighborsClassifier, MyNaiveBayesClassifier
import numpy as np
from tabulate import tabulate
import mysklearn.myevaluation as myevaluation

def remove_column(data,header,column_name):
    col_index = header.index(column_name)
    for row in data:
        row.pop(col_index)
    header.pop(col_index)
    return data, header

def get_column(data,header,column_name):
    col_index = header.index(column_name)
    col_data = []
    for row in data:
        col_data.append(row[col_index])
        
    return col_data


def get_groups_in_col(data,header,column_name):
    group_labels = []
    col_index = header.index(column_name)
    for row in data:
        if row[col_index] not in group_labels:
            group_labels.append(row[col_index])
    return group_labels

def remove_gender_rows(data,column,index):
    for i,row in enumerate(column):
        if row == "Female" or row == "F" or row == "female" or row == "f" or row == "Cis Female" or row == "Female (cis)":
            data[i][index] = "F"
        elif row == "Male" or row == "M" or row == "male" or row == "m" or row == "Male (CIS)" or row == "cis male":
            data[i][index] = "M"
        else:
            column.pop(i)
            data.pop(i)
    return data

def remove_NA(data,header,column_name):
    col_index = header.index(column_name)
    for row in data:
        if row[col_index] == "NA":
            data.pop(col_index)
    return data

def group_by_age(data,header,column_name):
    """Group by age. Based on the American Marketing Association Classification"""
    group_labels = ["21 and under","22 to 34","35 to 44","45 to 54","55 to 64","65 and over"]
    col_index = header.index(column_name)
    for row in data:
        if row[col_index] <= 21.0:
            row[col_index] = group_labels[0]
        elif row[col_index] >= 22.0 and row[col_index] <= 34.0:
            row[col_index]= group_labels[1]
        elif row[col_index] >= 35.0 and row[col_index] <= 44.0:
            row[col_index] = group_labels[2]
        elif row[col_index] >= 45.0 and row[col_index] <= 54.0:
            row[col_index] = group_labels[3]
        elif row[col_index] >= 55.0 and row[col_index] <= 64.0:
            row[col_index] = group_labels[4]
        elif row[col_index] >= 65.0:
            row[col_index] = group_labels[5]
        else:
            print("invalid age")
    return data

def convert_attributes_to_numeric(data,header):
    for i,col in enumerate(header):
        labels = get_groups_in_col(data,header,col)
        for row in data:
            idx = labels.index(row[i])
            row[i] = idx
    return data

def col_pval(p_values, header):
    new_header = []
    print("p-values > 0.05")
    for p_val in range(len(p_values)):
        if p_values[p_val] > 0.05:
            print(header[p_val], str(p_values[p_val]))
            new_header.append(header[p_val])

    return new_header

def parameterized_classifiers(train_folds,test_folds,n_splits,survey_table,y_data,classifier_name):
    classifier_names = ["Dummy Classifier","KNN Classifier","Naive Bayes Classifier","Decision Tree Classifier"]
    classifier_obj = [MyDummyClassifier(),MyKNeighborsClassifier(),MyNaiveBayesClassifier(),MyDecisionTreeClassifier()]
    all_X_train = []
    all_y_train = []
    all_y_test = []
    all_X_test = []
    all_y_predict = []

    for i in range(n_splits):
        X_train = []
        y_train  = []
        X_test = []
        y_test = []

        for j in train_folds[i]:
            X_train.append(survey_table.data[j])
            y_train.append(y_data[j])
        for j in test_folds[i]:
            X_test.append(survey_table.data[j])
            y_test.append(y_data[j])       
        all_X_train += X_train
        all_y_train += y_train
        all_X_test += X_test
        all_y_test += y_test

    if classifier_name == "Decision Tree Classifier":
        all_X_train.insert(0,survey_table.column_names)
        all_y_train.insert(0,"mental health consequence")

    classifier_idx = classifier_names.index(classifier_name)
    classifier = classifier_obj[classifier_idx]
    classifier.fit(all_X_train, all_y_train)
    all_y_predict = classifier.predict(all_X_test)
    predicted_instances = []
    for i in all_y_predict:
        predicted_instances.append(i[0])
    all_y_predict = predicted_instances

    total_accuracy = myevaluation.accuracy_score(all_y_test, all_y_predict)
    total_error = 1 - total_accuracy

    strat_accuracy = float("{:.2f}".format(total_accuracy))
    strat_error = float("{:.2f}".format(total_error))
    
    # calls myclassifiers.py for precision, recall and f1
    precision = myevaluation.binary_precision_score(all_y_test,all_y_predict)
    recall = myevaluation.binary_recall_score(all_y_test,all_y_predict)
    f1 = myevaluation.binary_f1_score(all_y_test,all_y_predict)

    print(classifier_name,"(Stratified 10-Fold Cross Validation Results):")
    print("--------------------------------------------------------------")
    print("Accuracy:", strat_accuracy)
    print("Error Rate:", strat_error)
    print()
    print("Precision:", float("{:.2f}".format(precision)))
    print("Recall:", float("{:.2f}".format(recall)))
    print("F1:", float("{:.2f}".format(f1)))

    labels = ["No","Maybe","Yes"]
    matrix = myevaluation.confusion_matrix(all_y_test,all_y_predict,labels)
    confusion_matrix_printing(classifier_name,matrix)
        
def confusion_matrix_printing(classifier_name,matrix):
    labels = ["No","Maybe","Yes"]
    pos_label = "Yes"
    print()
    print("Confusion Matrices")
    print("===========================================")
    all_data = [[] for i in range(0,len(labels)+1)]
    print()
    print(classifier_name,"(Stratified 10-fold cross validation Results):")
    all_data[0].append("Mental Health Consequences")
    for label in labels:
        all_data[0].append(label)
    all_data[0].append("Total")
    all_data[0].append("Recognition (%)")

    for j,label in enumerate(labels):
        all_data[j+1].append(label)
        total = 0
        for val in matrix[j]:
            total += val
            all_data[j+1].append(str(val))
        if total > 0:
            recognition = matrix[j][j] / total *100
        else:
            recognition = 0
        all_data[j+1].append(total)
        all_data[j+1].append(recognition)

    table = tabulate(all_data,headers="firstrow",numalign="right")
    print(table)