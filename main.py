import pandas as pd
import numpy as np

dt=pd.read_csv('survey_lung_cancer.csv')
pd.set_option('display.max_columns', 16)
print(dt)

#display first few rows of Dataframe
print(dt.head())

#Display information about the DataFrame
print(dt.info())

# Shows summary statistics for all numerical columns
dt.describe()

#check for missing values
print(dt.isnull())

#count missing values in each column
print(dt.isnull().sum())

import matplotlib.pyplot as plt #creating graphs and visualizations
import seaborn as sns #making advanced and attractive statistical plots

#bar plot for respondent gender
colors = ['skyblue', 'lightpink']
dt['GENDER'].value_counts().plot(kind='bar', color=colors)
plt.title("Gender")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()

# Count the Yes/No responses
counts = dt['LUNG_CANCER'].value_counts()

# Pie chart
plt.pie(
    counts, 
    labels=counts.index,              
    autopct='%1.1f%%',
    colors=['lightcoral', 'lightskyblue'],
    startangle=90,
    explode=[0.1, 0]
)

plt.title("Respondents Affected by Lung Cancer")
plt.axis('equal')
plt.show()

# Identify and Handle missing values
dt_clean = dt.copy()
dt_clean = dt_clean.fillna(dt_clean.mode().iloc[0])
dt.isnull().sum()

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
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Min-Max Normalization
scaler = MinMaxScaler()
dt_clean[num_cols] = scaler.fit_transform(dt_clean[num_cols])
print(dt_clean[num_cols].head())

# Standardization (Z-score)
scaler_std = StandardScaler()
dt_clean[num_cols] = scaler_std.fit_transform(dt_clean[num_cols])
print(dt_clean[num_cols].head())

#Age Binning
dt_clean['AGE_GROUP'] = pd.cut(
        dt_clean['AGE'],
        bins=[0, 30, 50, 100],
        labels=['Young', 'Adult', 'Senior']
    )
print(dt)

#Create indicator variables(one-hot encoding)
dt_final = pd.get_dummies(dt_clean, drop_first=True)
