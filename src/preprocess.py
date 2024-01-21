import pandas as pd
import numpy as np

import pandas_flavor as pf

# fungsi .clean_names()
#   .remove_empty()

@pf.register_dataframe_method
def change_empty_space_to_underscore(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Change empty space to underscore in dataframe column

    Parameters:
    -----------
    df : DataFrame object
    
    Returns:
    -----------
    df : DataFrame object
        The dataframe that has been changed

    """
    df.columns = df.columns.str.replace(' ', '_').str.lower()
    return df

@pf.register_dataframe_method
def fillna_with_zero_for_numeric(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Fill 0 for missing values in numeric columns

    Parameters:
    -----------
    df : DataFrame object
    
    Returns:
    -----------
    df : DataFrame object
        The dataframe that has been changed

    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    cols_to_fill = df.select_dtypes(include=numerics).columns
    fill_dict = {col : 0 for col in cols_to_fill}
    df = df.fillna(fill_dict)
    df[cols_to_fill] = df[cols_to_fill].astype(int)
    return df

'''credit: https://github.com/agostontorok/tdd_data_analysis/blob/master/data_analysis.py'''
@pf.register_dataframe_method
def vectorize_categorical_columns(
    df: pd.DataFrame
) -> pd.DataFrame:

    # need to change to dynamic categorical to change custom
    cat_features = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']

    df = pd.concat([df, pd.get_dummies(df[cat_features],
                               drop_first=True)], axis=1).reset_index(drop=True)
    
    df = df.drop(columns=cat_features)
    return df