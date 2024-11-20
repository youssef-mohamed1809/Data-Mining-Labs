import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

def data_clean(data):
    numeric_features  = data.select_dtypes(exclude=['object']).columns.tolist()
    numeric_data = data[numeric_features]
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp_mean.fit(numeric_data)
    numeric_data = imp_mean.transform(numeric_data)
    data[numeric_features] = numeric_data
    return data

def numeric_features_normalize(data):
    numeric_features  = data.select_dtypes(exclude=['object']).columns.tolist()
    numeric_data = data[numeric_features]
    normalizer = MinMaxScaler(feature_range=(0,1))
    norm_data = normalizer.fit_transform(numeric_data)
    data[numeric_features] = norm_data
    return data

if __name__ == '__main__':
    np.set_printoptions(suppress=True, formatter={'float_kind': lambda x: f'{x:.0f}'})
    DataFrame = pd.DataFrame({
        'Age': [49, 32, 35, 43, 45, 40, np.nan],
        'Income': [86400, 57600, 64800, 73200, np.nan, 69600, 62400]
    })
    DataFrame = data_clean(DataFrame)
    DataFrame = numeric_features_normalize(DataFrame)
    print(DataFrame)