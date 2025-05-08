# preprocessing.py

import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_input():
    df = pd.read_csv("Test.pcap_ISCX.csv")
    df.columns = df.columns.str.strip()
    df = df.replace([float('inf'), float('-inf')], pd.NA).dropna()
    df['Label'] = df['Label'].apply(lambda x: 1 if 'DoS' in x else 0)

    X = df.drop(columns=['Label'])
    y = df['Label']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y
