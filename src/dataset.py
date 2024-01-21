# Load Dataset, define which one is X/features, which is Y/target variable
# Cleaning

# https://github.com/agostontorok/tdd_data_analysis/blob/master/data_analysis.py
# https://github.com/tirthajyoti/Machine-Learning-with-Python/tree/master/OOP_in_ML

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from supervised.automl import AutoML

from sklearn.metrics import accuracy_score

from typing import (
    Tuple,
    Iterable
)

from src.utils import impute, flag_nulls
from src.preprocess import change_empty_space_to_underscore, fillna_with_zero_for_numeric, vectorize_categorical_columns

CREDIT_RISK_DATASET = '../data/raw/credit_risk_dataset.csv'

def load_data(
    dataset: str
)-> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    dataset = {
        'credit_risk': {
            'data': CREDIT_RISK_DATASET,
            'labels': 'loan_status',
            'missing_value_handling': 'fill_zero'
        },
        'other_dataset': {
            'data': CREDIT_RISK_DATASET,
            'labels': 'loan_status',
            'missing_value_handling': 'fill_average'
        }
    }[dataset]

    df = pd.read_csv(dataset['data'])

    return df, dataset['labels'], dataset['missing_value_handling']

def clean_data(
    dataset: str,
    df: pd.DataFrame,
    labels: str,
    missing_value_handling: str
) -> pd.DataFrame:

    df = df.change_empty_space_to_underscore()

    df = df.vectorize_categorical_columns()

    match missing_value_handling:
        case 'fill_zero':
            match dataset:
                case 'credit_risk':
                    df = (
                        df.flag_nulls()
                            .impute(column_name='person_emp_length', value=0)
                            .impute(column_name='loan_int_rate', value=0)
                    )
        case 'fill_average':
            match dataset:
                case 'credit_risk':
                    df = (
                        df.flag_nulls()
                            .impute(column_name='person_emp_length', statistic_column_name='mean')
                            .impute(column_name='loan_int_rate', statistic_column_name='mean')
                    )
        case default:
            match dataset:
                case 'credit_risk':
                    df = (
                        df.flag_nulls()
                            .impute(column_name='person_emp_length', value=0)
                            .impute(column_name='loan_int_rate', value=0)
                    )
    return df

if __name__ == '__main__':
    df, labels, missing_value_handling = load_data(dataset='credit_risk')
    
    # to avoid error in shape
    df = df.drop_duplicates(keep='last')

    # split data before filling NaN & vectorizing to avoid data leak
    # stratify to make sure we get all of the categorical variables on train set
    X = df.drop(columns=labels)
    y = df[labels]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=46)
    # loan_grade can't be stratified since it would make error

    X_train = clean_data('credit_risk', X_train, labels, missing_value_handling)
    X_test = clean_data('credit_risk', X_test, labels, missing_value_handling)
    print(X_train.shape[0])
    print(y_train.shape[0])

    # autoML
    #cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    automl = AutoML()
    automl.fit(X_train, y_train)

    predictions = automl.predict(X_test)
    
    print(y_test)
    print(accuracy_score(y_test, predictions.astype(int)))
