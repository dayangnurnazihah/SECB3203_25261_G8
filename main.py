import pandas as pd
import numpy as np

dt=pd.read_csv('survey_lung_cancer.csv')
pd.set_option('display.max_columns', 16)
print(dt)

# Handle missing values
dt_clean = dt.copy()
dt_clean = dt_clean.fillna(dt_clean.mode().iloc[0])

#Format data (Convert Yes/No to 1/0, fix data types)
#Convert Yes/No to 1/0
yes_no_cols = [col for col in dt_clean.columns if dt_clean[col].dtype == 'object']

for col in yes_no_cols:
    dt_clean[col] = dt_clean[col].replace({
        'YES': 1, 'NO': 0,
        'Yes': 1, 'No': 0,
        'Y': 1, 'N': 0
    })
  
# Fix data type
dt_clean = dt_clean.apply(pd.to_numeric, errors='ignore')

#Normalize numerical data (scalling/centering)
from sklearn.preprocessing import MinMaxScaler

num_cols = dt_clean.select_dtypes(include=['int64', 'float64']).columns
scaler = MinMaxScaler()
dt_clean[num_cols] = scaler.fit_transform(dt_clean[num_cols])

#Age Binning
if 'AGE' in dt_clean.columns:
    dt_clean['AGE_GROUP'] = pd.cut(
        dt_clean['AGE'],
        bins=[0, 30, 50, 100],
        labels=['Young', 'Adult', 'Senior']
    )

#Create indicator variables(one-hot encoding)
dt_final = pd.get_dummies(dt_clean, drop_first=True)
