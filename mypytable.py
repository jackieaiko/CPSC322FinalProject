#############################################
#Programmer: Lin Ai Tan
#Class: CPSC 322-01, Spring 2021
#Programming Assignment #2
#1/26/21
#I attempted the pylint bonus task and allow two input files

#Description: This program has all the functions needed to clean or find information about a dataset
#perform som esimple analysis
#################################################

"""copy and csv module"""
import copy
import csv

# required functions/methods are noted with TODOs
# provided unit tests are in test_mypytable.py
# do not modify this class name, required function/method headers, or the unit tests
class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    # def pretty_print(self):
    #     """Prints the table in a nicely formatted grid structure.
    #     """
    #     print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        rows = len(self.data)
        cols = len(self.column_names)
        return rows,cols

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Notes:
            Raise ValueError on invalid col_identifier
        """
        ret_column = []
        col_index = -1
        for i,name in enumerate(self.column_names):
            if col_identifier in (i,name):
                col_index = i
                break
        for rows in self.data:
            if rows[col_index] == "NA":
                if include_missing_values:
                    ret_column.append(rows[col_index])
            else:
                ret_column.append(rows[col_index])
        return ret_column

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        for i,row in enumerate(self.data):
            for col in range(len(self.data[i])):
                try:
                    row[col] = float(row[col])
                except ValueError:
                    row[col] = str(row[col])

    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.

        Args:
            row_indexes_to_drop(list of int): list of row indexes to remove from the table data.
        """
        for i in reversed(row_indexes_to_drop):
            del self.data[i]

    @classmethod
    def split_lines(cls,line):
        """Splits a single line of a csv file by checking to make sure each column is separated
        by ',' stored as a seperate element in a list.Prevents splitting by ','
        if the film title has ""
        Args:
            line: str of csv rows where cols are separated by ,
        Returns:
        a list: properly seperated line by their comma except for commas surrounded by
        double quotes
        """
        first = -1
        words = []
        trim_string = ""
        for i, char in enumerate(line):
            if char == '"':
                if first == -1:
                    first = i
                else:
                    first = -1
                trim_string += char
            else:
                if char == ',':
                    if first == -1:
                        words.append(trim_string)
                        trim_string = ""
                    else:
                        trim_string += char
                else:
                    trim_string += char
        words.append(trim_string)
        return words

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Args:
            filename(str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: return self so the caller can write code like
                table = MyPyTable().load_from_file(fname)

        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        with open(filename, "r") as file:
            reader = csv.reader(file)
            rownum = 0
            for row in reader:
                if rownum == 0:
                    header = row
                    for col in header:
                        self.column_names.append(str(col))
                else:
                    curr_row = []
                    for col in row:
                        curr_row.append(str(col))
                    self.data.append(curr_row)
                rownum += 1
        self.convert_to_numeric()
        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """
        # writing to csv file
        with open(filename, 'w', newline = '\n') as csvfile:
            # creating a csv writer object
            csvwriter = csv.writer(csvfile)
            # writing the fields
            csvwriter.writerow(self.column_names)
            # writing the data rows
            csvwriter.writerows(self.data)

    def find_duplicates(self, key_column_names):
        """Returns a list of indexes representing duplicate rows.
        Rows are identified uniquely based on key_column_names.

        Args:
            key_column_names(list of str): column names to use as row keys.

        Returns
            list of int: list of indexes of duplicate rows found

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
                The first instance of a row is not considered a duplicate.
        """
        hashtable = []
        duplicates = []
        for i,row in enumerate(self.data):
            value = ""
            for col_name in key_column_names:
                col_index = self.column_names.index(col_name)
                value += str(row[col_index])
            if value in hashtable:
                duplicates.append(i)
            else:
                hashtable.append(value)
        return duplicates

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """
        for i, row in enumerate(self.data):
            mark_rm = False
            for col in row:
                if col == "NA":
                    mark_rm = True
            if mark_rm:
                self.drop_rows([i])
        #return self.data

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
            by the column's original average.

        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        #self.convert_to_numeric()
        col_index = self.column_names.index(col_name)
        total_sum = 0
        num_rows = 0
        for row in self.data:
            if row[col_index] != "NA" and isinstance(row[col_index], float):
                total_sum += row[col_index]
                num_rows +=1
        avg = total_sum / num_rows
        for row in self.data:
            if row[col_index] == "NA":
                row[col_index] = avg

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.

        Args:
            col_names(list of str): names of the continuous columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]
        """
        self.remove_rows_with_missing_values()
        #self.convert_to_numeric()
        results = []
        result_col_names = ["attribute", "min", "max", "mid", "avg", "median"]
        d_min = 0
        d_max = 0
        d_avg = 0
        d_mid = 0
        d_median = 0
        for col in col_names:
            values = []
            result = []
            col_index = self.column_names.index(col)
            for row in self.data:
                if row[col_index] != "NA":
                    values.append(row[col_index])
            values.sort()
            if len(values) > 0:
                d_min = values[0]
                d_max = values[-1]
                d_mid = (d_max + d_min)/2
                d_avg = sum(values)/len(values)
                if len(values) %2 == 0: #even number of elements
                    d_median = (values[(len(values)//2)-1] + values[((len(values)//2) +1)-1])/2
                else:
                    d_median = values[((len(values)+1)//2)-1]
                result.append(col)
                result.append(d_min)
                result.append(d_max)
                result.append(d_mid)
                result.append(d_avg)
                result.append(d_median)
                results.append(result)
        return MyPyTable(result_col_names,results)

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
            with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """
        num = 0
        new_columns = []
        new_table = []
        new_columns = copy.deepcopy(self.column_names)
        for col_name in other_table.column_names:
            if col_name not in new_columns:
                new_columns.append(col_name)
        for row_orig_data in self.data:
            for row_other_data in other_table.data:
                join = True
                new_row = []
                for col_name in key_column_names:
                    col_index1 = self.column_names.index(col_name)
                    col_index2 = other_table.column_names.index(col_name)
                    if row_orig_data[col_index1] != row_other_data[col_index2]:
                        # mark as don't add
                        join = False
                if join:
                    new_row = self.build_row(row_orig_data,row_other_data,new_row)
                    new_table.append(new_row)
                    num += 1
        return MyPyTable(new_columns,new_table)

    def perform_full_outer_join(self,other_table,key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with
            other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pad the attributes with missing values with "NA".
        """
        new_columns = []
        new_table = []
        new_columns = copy.deepcopy(self.column_names)
        new_columns = self.load_columns(other_table,new_columns)
        for row_orig_data in self.data:
            row_join = False
            for row_other_data in other_table.data:
                join = True
                new_row = []
                for col in row_orig_data:
                    new_row.append(col)
                for col_name in key_column_names:
                    col_index1 = self.column_names.index(col_name)
                    col_index2 = other_table.column_names.index(col_name)
                    if row_orig_data[col_index1] != row_other_data[col_index2]:
                        # mark as don't add
                        join = False
                if join:
                    for col in row_other_data:
                        if col not in new_row:
                            new_row.append(col)
                    row_join = True
                    new_table.append(new_row)
            if not row_join:
                for col_name in other_table.column_names:
                    if col_name not in self.column_names:
                        new_row.append("NA")
                new_table.append(new_row)
        self.iterate_remaining(new_table,other_table,new_columns)
        return MyPyTable(new_columns,new_table)

    @classmethod
    def iterate_remaining(cls,new_table,other_table,new_columns):
        """Return a new table containes the other_table that has no match with the original table

        Args:
            other_table(MyPyTable): the second table to join this table with.
            new_columns(list of str): column names of new table
        Returns:
            new_table and new_columns
        Notes:
            Pad the attributes with missing values with "NA".
        """
        for row_other_data in other_table.data:
            empty = True
            for rows in new_table:
                if set(row_other_data).issubset(set(rows)):
                    empty = False
            if empty:
                new_row = []
                for col_name in new_columns:
                    if col_name not in other_table.column_names:
                        new_row.append("NA")
                    else:
                        col_index = other_table.column_names.index(col_name)
                        new_row.append(row_other_data[col_index])
                new_table.append(new_row)
        return new_columns

    @classmethod
    def load_columns(cls,other_table,new_columns):
        """Return a new_column that has all the attributes from the original and other table.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            new_columns(list of str): column names.

        Returns:
            new_columns of the new table
        """
        for col_name in other_table.column_names:
            if col_name not in new_columns:
                new_columns.append(col_name)
        return new_columns

    @classmethod
    def build_row(cls,row_orig_data, row_other_data,new_row):
        """Builds a single row by combining unique values in both rows from two tables

        Args:
            row_orig_data(list or mix str and int) represent a single row in original data
            row_other_data(list or mix str and int) represent a single row in other table

        Returns:
            a 1D list of str or int of newly combined values from two list.
        """
        for col in row_orig_data:
            new_row.append(col)
        for col in row_other_data:
            if col not in new_row:
                new_row.append(col)
        return new_row
