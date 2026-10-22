import pandas as pd
from sklearn.model_selection import train_test_split

#def read_training_data(file_path):
#    df = pd.read_csv(file_path)
#    return df

def train_test_split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

    
def treat_missing_values(df, cat_cols, num_cols):
    # categorical columns will be assigned a placeholder value (e.g., 'missing') before encoding
    for col in cat_cols:
        df[col] = df[col].fillna('missing')
        df[col] = df[col].astype('category')
    # numerical columns will be filled with the median value
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    return df

    return 