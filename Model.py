import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


pd.set_option('display.max_columns', 500)

df = pd.read_csv("AGGREGATED_INSTITUTION_DATA.csv").dropna(axis=0, inplace=False)
year = df['YEAR']
instnm_table = df[['UNITID', 'INSTNM']]
data = df.select_dtypes(include=['number'])
data['YEAR'] = year

schools = df['UNITID'].unique()
schools.sort()

columns_to_predict = {'UNITID': [], 'ADM_RATE': [], 'SATVRMID': [], 'SATMTMID': [], 'ACTCMMID': [], 'ACTCM25': [], 'ACTCM75': [], 'COSTT4_A': [],
                      'UGDS': [], 'C150_4': [], 'RET_FT4': [], 'MD_EARN_WNE_MALE0_P6': [], 'MD_EARN_WNE_MALE1_P6': [],
                      'MD_EARN_WNE_MALE0_P10': [], 'MD_EARN_WNE_MALE1_P10': []}

for school in schools:
    # add_ID = columns_to_predict.get('UNITID')
    # add_ID.append(school)
    # columns_to_predict['UNITID'] = add_ID
    for column in list(columns_to_predict.keys()):
        X = data[data['UNITID'] == school]
        X = X[['YEAR', column]]
        if len(X) == 1:
            to_add = columns_to_predict.get(column)
            to_add.append(X[column].iloc[0])
            columns_to_predict[column] = to_add
        else:
            predicted_val = list(X[column])[-1] + (list(X[column])[-1] - list(X[column])[-2])
            y = ['2023_24', predicted_val]
            regression = LinearRegression().fit(X, y)
            predicted = regression.predict(X)
            to_add = columns_to_predict.get(column)
            to_add.append(predicted[1])
            columns_to_predict[column] = to_add

columns_to_predict['UNITID'] = schools
final_data = pd.DataFrame(columns_to_predict)
final_data.to_csv("C:/Users/Dials/PycharmProjects/CS490CapstoneModel/predicted_values.csv")
