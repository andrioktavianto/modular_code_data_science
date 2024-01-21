# useful function to preprocess data

# def get_row(column, row_value). Row value can be >, <, = e.g. <= 5. Tujuannya mempersingkat 
# biar gak perlu ngetik df[df['column'] == 5] atau df.loc[df['column'] == 5]

import pandas as pd
import numpy as np
from scipy.stats import mode

import pandas_flavor as pf
from typing import (
    Dict,
    Callable,
    Iterable,
    Union,
    Any, 
    Hashable, 
    Optional
)

'''Credit: https://github.com/pyjanitor-devs/pyjanitor/blob/f65a289b3892b13101955048a9bf026c875a5427/janitor/functions/rename_columns.py'''
@pf.register_dataframe_method
def rename_column(
    df: pd.DataFrame
    , old_column_name: str
    , new_column_name: str
) -> pd.DataFrame:
    """
    Rename a column in dataframe.

    Example usage:
    df.rename_column(old_column_name='a', new_column_name='a_new')

    Parameters:
    -----------
    df : DataFrame object
    
    old_column_name : str
        The old column name
    
    new_column_name : str
        The new column name
    
    Returns:
    -----------
    df : DataFrame object
        The dataframe that has been changed

    """
    check_column(df, [old_column_name])

    return df.rename(columns={old_column_name: new_column_name})

'''Credit: https://github.com/pyjanitor-devs/pyjanitor/blob/f65a289b3892b13101955048a9bf026c875a5427/janitor/functions/rename_columns.py'''
@pf.register_dataframe_method
def rename_columns(
    df: pd.DataFrame
    , new_column_names: Union[Dict, None] = None
    , function: Callable = None
) -> pd.DataFrame:

    """
    Rename multiple columns in dataframe.

    Example usage:
    df.rename_columns(new_column_names={"a": "a_new", "b": "b_new"})
    df.rename_columns(function=str.upper)

    Parameters:
    -----------
    df : DataFrame object
    
    new_column_names : dictionary
        The old & new columns name in dictionary

    function: Python function
        A function which should be applied to all the columns.
    
    Returns:
    -----------
    df : DataFrame object
        The dataframe that has been changed

    """

    if new_column_names is None and function is None:
        raise ValueError(
            "One of new_column_names or function must be provided"
        )

    if new_column_names is not None:
        check_column(df, new_column_names)
        return df.rename(columns=new_column_names)

    return df.rename(mapper=function, axis="columns")

'''Credit: https://github.com/pyjanitor-devs/pyjanitor/blob/f65a289b3892b13101955048a9bf026c875a5427/janitor/functions/rename_columns.py'''
def check_column(
    df: pd.DataFrame
    , column_names: Union[Iterable, str]
    , present: bool = True
):
    """
    One-liner syntactic sugar for checking the presence or absence
    of columns.

    Example usage:

    ```python
    check_column(df, ['a', 'b'], present=True)
    ```

    This will check whether columns `'a'` and `'b'` are present in
    `df`'s columns.

    One can also guarantee that `'a'` and `'b'` are not present
    by switching to `present=False`.

    Parameters:
    -----------
    df: DataFrame Object
        Text containing monetary value.

    column_names: str
        A list of column names we want to check to see if
        present (or absent) in `df`
    
    present: boolean
        If `True` (default), checks to see if all of `column_names`
        are in `df.columns`. If `False`, checks that none of `column_names` are
        in `df.columns`
    
    Returns:
    -----------
    ValueError: if data is not the expected type.

    """
    if isinstance(column_names, str) or not isinstance(column_names, Iterable):
        column_names = [column_names]

    for column_name in column_names:
        if present and column_name not in df.columns: 
            raise ValueError(
                f"{column_name} not present in dataframe columns!"
            )
        elif not present and column_name in df.columns:
            raise ValueError(
                f"{column_name} already present in dataframe columns!"
            )

'''Credit: https://github.com/pyjanitor-devs/pyjanitor/blob/a4fdff663e5a4590225749391db1ab60edd985fa/janitor/functions/impute.py'''
@pf.register_dataframe_method
def impute(
    df: pd.DataFrame,
    column_name: Hashable,
    value: Optional[Any] = None,
    statistic_column_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Impute value to a column

    Example usage:

    ```python
    # Impute null values with 0
    .impute(column_name='sales', value=0.0)
            
    # Impute null values with median
    .impute(column_name='score', statistic_column_name='median')
    ```

    Parameters:
    -----------
    df: DataFrame Object
        Text containing monetary value.

    column_names: str
        A list of column names we want to check to see if
        present (or absent) in `df`
    
    value: int, float, or str (optional)
        The value to impute

    statistic_column_name: `mean`, `average` `median`, `mode`, 
                            `minimum`, `min`, `maximum`, or `max` (optional)
        The column statistic to impute
    
    Returns:
    -----------
    ValueError: if data is not the expected type.

    """

    # Firstly, we check that only one of `value` or `statistic` are provided.
    if value is not None and statistic_column_name is not None:
        raise ValueError(
            "Only one of `value` or `statistic` should be provided"
        )

    # If statistic is provided, then we compute the relevant summary statistic
    # from the other data.
    funcs = {
        "mean": np.mean,
        "average": np.mean,  # aliased
        "median": np.median,
        "mode": mode,
        "minimum": np.min,
        "min": np.min,  # aliased
        "maximum": np.max,
        "max": np.max,  # aliased
    }
    if statistic_column_name is not None:
        # Check that the statistic keyword argument is one of the approved.
        if statistic_column_name not in funcs.keys():
            raise KeyError(f"`statistic` must be one of {funcs.keys()}")

        value = funcs[statistic_column_name](
            df[column_name].dropna().to_numpy()
        )
        # special treatment for mode, because scipy stats mode returns a
        # moderesult object.
        if statistic_column_name == "mode":
            value = value.mode[0]

    # The code is architected this way - if `value` is not provided but
    # statistic is, we then overwrite the None value taken on by `value`, and
    # use it to set the imputation column.
    if value is not None:
        df[column_name] = df[column_name].fillna(value)
    return df

'''Credit: https://github.com/pyjanitor-devs/pyjanitor/blob/f65a289b3892b13101955048a9bf026c875a5427/janitor/functions/flag_nulls.py'''
@pf.register_dataframe_method
def flag_nulls(
    df: pd.DataFrame,
    column_name: Optional[Hashable] = "null_flag",
    columns: Optional[Union[str, Iterable[str], Hashable]] = None,
) -> pd.DataFrame:
    """Creates a new column to indicate whether you have null values in a given
    row. If the columns parameter is not set, looks across the entire
    DataFrame, otherwise will look only in the columns you set.
    This method does not mutate the original DataFrame.
    Example:
        >>> import pandas as pd
        >>> import janitor
        >>> df = pd.DataFrame({
        ...     "a": ["w", "x", None, "z"], "b": [5, None, 7, 8],
        ... })
        >>> df.flag_nulls()
              a    b  null_flag
        0     w  5.0          0
        1     x  NaN          1
        2  None  7.0          1
        3     z  8.0          0
        >>> df.flag_nulls(columns="b")
              a    b  null_flag
        0     w  5.0          0
        1     x  NaN          1
        2  None  7.0          0
        3     z  8.0          0
    :param df: Input pandas DataFrame.
    :param column_name: Name for the output column.
    :param columns: List of columns to look at for finding null values. If you
        only want to look at one column, you can simply give its name. If set
        to None (default), all DataFrame columns are used.
    :returns: Input dataframe with the null flag column.
    :raises ValueError: if `column_name` is already present in the
        DataFrame.
    :raises ValueError: if any column within `columns` is not present in
        the DataFrame.
    """
    # Sort out columns input
    if isinstance(columns, str):
        columns = [columns]
    elif columns is None:
        columns = df.columns
    elif not isinstance(columns, Iterable):
        # catches other hashable types
        columns = [columns]

    # Input sanitation checks
    check_column(df, columns)
    check_column(df, [column_name], present=False)

    # This algorithm works best for n_rows >> n_cols. See issue #501
    null_array = np.zeros(len(df))
    for col in columns:
        null_array = np.logical_or(null_array, pd.isna(df[col]))

    df = df.copy()
    df[column_name] = null_array.astype(int)
    return df