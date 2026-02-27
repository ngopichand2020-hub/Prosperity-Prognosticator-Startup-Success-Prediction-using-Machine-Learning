import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(path):

    df = pd.read_csv(path)

    return df


def clean_data(df):

    # Drop unnecessary columns
    drop_cols = [
        'Unnamed: 0',
        'Unnamed: 6',
        'object_id',
        'name',
        'state_code.1',
        'zip_code',
        'id'
    ]

    df = df.drop(columns=drop_cols, errors='ignore')

    # Fill missing values
    df = df.fillna(0)

    return df


def encode_data(df):

    le = LabelEncoder()

    for col in df.select_dtypes(include='object'):

        df[col] = df[col].astype(str)

        df[col] = le.fit_transform(df[col])

    return df


def split_data(df):

    X = df.drop("status", axis=1)

    y = df["status"]

    return X, y
def select_features(df):

    important_features = [
        'funding_total_usd',
        'funding_rounds',
        'relationships',
        'milestones',
        'avg_participants',
        'is_top500',
        'has_VC',
        'age_first_funding_year'
    ]

    X = df[important_features]

    y = df["status"]

    return X, y
