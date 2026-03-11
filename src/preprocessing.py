import pandas as pd
import numpy as np

# Load dataset
def load_data(path):
    df = pd.read_csv(path)
    return df

# Handle missing values using country-wise mean
def handle_missing_values(df):
    country_means = df.groupby("Country").transform("mean")
    df = df.fillna(country_means)
    return df

# Remove duplicate rows
def remove_duplicates(df):
    df = df.drop_duplicates()
    return df

# Remove outliers using IQR
def remove_outliers_iqr(df, columns):

    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)

        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    return df