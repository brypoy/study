######## STANDARD IMPORT ##############################################################
import sys
import os
sys.path.append(os.path.expanduser('~/Projects/bry_mod/'))
from bry_mod import *

import sys
import os
sys.path.append(os.path.expanduser('~/Projects/bry_mod/'))
from bry_mod import *

#print(bry_mod_test_string)
#######################################################################################
#https://pandas.pydata.org/docs/user_guide/10min.html

def create_initial_table():
    # Creating date range
    dates = pd.date_range(start='2024-01-01', periods=7)

    # Creating numerical and alpha columns
    data = {
        'Date': dates,
        'Numeric_Column1': [10, 20, 30, 40, 50, 60, 70],
        'Numeric_Column2': [5.5, 6.7, 8.9, 10.1, 12.3, 14.5, 16.7],
        'Alpha_Column1': ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
        'Alpha_Column2': ['apple', 'banana', 'carrot', 'dog', 'elephant', 'frog', 'giraffe'],
        'Grouper_Column': ['Group1', 'Group2', 'Group3', 'Group1', 'Group2', 'Group3', 'Group1']
    }

    # Creating DataFrame
    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)  # Set 'Date' column as index
    #df = pd.DataFrame(data, index=dates)
    save(df)
    return df


def use_of_head_tail_describe_T():
    heads = df.head(2)
    save(heads)
    #
    tails = df.tail(2)
    save(tails)
    #
    described = df.describe()
    save(described)
    #
    transposed = df.T
    save(transposed)


def get_rows_and_columns():
    # note the dataframe must be created with index for this to work: df = pd.DataFrame(data, index=dates)
    specific_index_and_columns = df.loc["20240101":"20240103", ["Numeric_Column2", "Alpha_Column1"]]
    save(specific_index_and_columns)
    #
    # here we set dates to the index so that it may be used in the loc function
    dates = df.index
    scalar_index_named_row = df.loc[dates[0:3], ["Numeric_Column2", "Alpha_Column1"]]
    save(scalar_index_named_row)


def group_and_trim():
    grouped_df = df.groupby('Grouper_Column')
    save(grouped_df)
    #
    ungrouped_df = pd.concat([group_data for _, group_data in grouped_df]).reset_index(drop=True)
    save(ungrouped_df)
    #
    ungrouped_two_df = pd.concat([group_data.reset_index(drop=True) for _, group_data in list(grouped_df)[:2]])
    save(ungrouped_two_df)
    #
    group1_data = grouped_df.get_group('Group1')
    save(group1_data)
    #



if __name__ == "__main__":
    df = create_initial_table()
    #use_of_head_tail_describe_T()
    #get_rows_and_columns()
    group_and_trim()
