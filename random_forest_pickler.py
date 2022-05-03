import pickle
from random import random 
from sklearn.feature_selection import chi2
from mypytable import MyPyTable 
import myutils
from copy import deepcopy
import os
from mysklearn.myclassifiers import MyKNeighborsClassifier, MyRandomForestClassifier
import mysklearn.myevaluation as myevaluation

survey_fname = os.path.join("input_data","survey.csv")
survey_table = MyPyTable()
survey_table.load_from_file(survey_fname)

#clean dataset
name_of_col_to_remove = ["Timestamp","Country","state","comments"]
for col_name in name_of_col_to_remove:
    survey_table.data, survey_table.column_names = myutils.remove_column(survey_table.data,survey_table.column_names,col_name)

#remove any gender label that is not Male, m, M, female, Female, f or Cis Female or Cis Male
gender_col= survey_table.get_column("Gender")
col_index = survey_table.column_names.index("Gender")
survey_table.data = myutils.remove_gender_rows(survey_table.data, gender_col,col_index)

# remove NA values
survey_table.data = myutils.remove_NA(survey_table.data,survey_table.column_names,"self_employed")
print(len(survey_table.data)) # only 9 rows removed
# remove NA values
survey_table.data = myutils.remove_NA(survey_table.data,survey_table.column_names,"work_interfere")
print(len(survey_table.data)) # 200 rows removed
# group age values
survey_table.data = myutils.group_by_age(survey_table.data,survey_table.column_names,"Age")

#att selection
copied_data = deepcopy(survey_table.data)
copied_column_names = deepcopy(survey_table.column_names)
copied_data = myutils.convert_attributes_to_numeric(copied_data,copied_column_names)

y_data = myutils.get_column(copied_data,copied_column_names, "mental_health_consequence")
X_data, X_header = myutils.remove_column(copied_data,copied_column_names, "mental_health_consequence")
chi_2, p_value = chi2(X_data, y_data)
headers_to_remove = myutils.col_pval(p_value, X_header)
y_data = myutils.get_column(survey_table.data,survey_table.column_names,"mental_health_consequence")

for header in headers_to_remove:
    survey_table.data, survey_table.column_names = myutils.remove_column(survey_table.data,survey_table.column_names,header)

y_data = myutils.get_column(survey_table.data,survey_table.column_names,"mental_health_consequence")
survey_table.data, survey_table.column_names = myutils.remove_column(survey_table.data,survey_table.column_names, "mental_health_consequence")
knn_classifier = MyKNeighborsClassifier()
knn_classifier.fit(survey_table.data,y_data)

# serialize to file (pickle)
outfile = open("knn.p", "wb")
pickle.dump(knn_classifier, outfile)
outfile.close()
