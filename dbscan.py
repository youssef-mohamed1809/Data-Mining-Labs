import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import DBSCAN

##nominal vales: sex, region, married, car, save_act, current_act

data = pd.read_csv("datasets/bank_dataset.csv")

# Remove id column
data = data.iloc[:,1:]


# Transform nominal data to numerical data
labelEncoder = LabelEncoder()

labelEncoder.fit(data['sex'])
data['sex'] = labelEncoder.transform(data['sex'])

labelEncoder.fit(data['region'])
data['region'] = labelEncoder.transform(data['region'])

labelEncoder.fit(data['married'])
data['married'] = labelEncoder.transform(data['married'])

labelEncoder.fit(data['car'])
data['car'] = labelEncoder.transform(data['car'])

labelEncoder.fit(data['save_act'])
data['save_act'] = labelEncoder.transform(data['save_act'])

labelEncoder.fit(data['current_act'])
data['current_act'] = labelEncoder.transform(data['current_act'])


# DBSCAN algorithm
dbscan = DBSCAN(eps=1.2, min_samples=3).fit(data)
print (dbscan.labels_)
