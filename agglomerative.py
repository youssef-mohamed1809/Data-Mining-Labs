from sklearn.cluster import AgglomerativeClustering
import pandas as pd
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("datasets/weather_nominal.csv")
labelEncoder = LabelEncoder()

labelEncoder.fit(data['outlook'])
data['outlook'] = labelEncoder.transform(data['outlook'])

labelEncoder.fit(data['temperature'])
data['temperature'] = labelEncoder.transform(data['temperature'])

labelEncoder.fit(data['humidity'])
data['humidity'] = labelEncoder.transform(data['humidity'])

labelEncoder.fit(data['windy'])
data['windy'] = labelEncoder.transform(data['windy'])

labelEncoder.fit(data['play'])
data['play'] = labelEncoder.transform(data['play'])


agg = AgglomerativeClustering().fit(data.iloc[:, 1:])

print(agg.labels_)