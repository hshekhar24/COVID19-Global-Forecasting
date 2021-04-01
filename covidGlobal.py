import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df_submission = pd.read_csv('submission.csv')

# process date
df['Date'] = pd.to_datetime(df['Date'])
df['Day'] = df['Date'].apply(lambda x: x.dayofyear).astype(np.int16)

df_test['Date'] = pd.to_datetime(df_test['Date'])
df_test['Day'] = df_test['Date'].apply(lambda x: x.dayofyear).astype(np.int16)

X = df.iloc[:, [ 3, 4,8 ]].values
X_res = df_test.iloc[:, [ 3, 4,6 ]].values
y = df.iloc[:, 6].values

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor.fit(X, y)

y_pred_case = regressor.predict(X_res)
y_pred_case = y_pred_case.astype(int)


X = df.iloc[:, [ 3, 4,8 ]].values
X_res = df_test.iloc[:, [ 3, 4,6 ]].values
y_fatality = df.iloc[:, 7].values   

from sklearn.ensemble import RandomForestRegressor
regressor2 = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor2.fit(X, y_fatality)

y_pred_fatality = regressor2.predict(X_res)
y_pred_fatality = y_pred_fatality.astype(int)

df_submission['ConfirmedCases']=y_pred_case
df_submission['Fatalities']=y_pred_fatality

