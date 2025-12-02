import pandas as pd
import numpy as np

dt=pd.read_csv('survey_lung_cancer.csv')
pd.set_option('display.max_columns', 16)
print(dt)