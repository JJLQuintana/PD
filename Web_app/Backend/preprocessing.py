import pandas as pd
from sklearn.preprocessing import StandardScaler

#data load and preprocessing

def preprocess_input():
    df = pd.read_csv("Web_app/Backend/Test.pcap_ISCX.csv") 
    df.columns = df.columns.str.strip()
    df = df.replace([float('inf'), float('-inf')], pd.NA).dropna() # dropping rows with nan and inf values
    df['Label'] = df['Label'].apply(lambda x: 1 if 'DoS' in x else 0)

    X = df.drop(columns=['Label'])
    y = df['Label']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  
    
    return X_scaled, y