from calendar import c


def remove_column(data,header,column_name):
    col_index = header.index(column_name)
    for row in data:
        row.pop(col_index)
    header.pop(col_index)
    return data, header

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
