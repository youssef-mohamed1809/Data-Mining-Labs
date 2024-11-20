import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def remove_outliers(data_frame):
    x1 = data_frame['area'].values
    x1_q1 = np.percentile(x1, 25)
    x1_q3 = np.percentile(x1, 75)
    x1_iqr = x1_q3 - x1_q1
    max_x1 = x1_q3 + (1.5 * x1_iqr)
    min_x1 = x1_q1 - (1.5 * x1_iqr)
    indexes = []
    for i in range(len(x1)):
        if x1[i] > max_x1 or x1[i] < min_x1:
            indexes.append(i)
    new_df = data_frame.drop(index=indexes)

    x2 = data_frame['rooms'].values
    x2_q1 = np.percentile(x2, 25)
    x2_q3 = np.percentile(x2, 75)
    x2_iqr = x2_q3 - x2_q1
    max_x2 = x2_q3 + (1.5 * x2_iqr)
    min_x2 = x2_q1 - (1.5 * x2_iqr)

    indexes = []
    for i in range(len(x2)):
        if x2[i] > max_x2 or x2[i] < min_x2:
            indexes.append(i)
    new_df = new_df.drop(index=indexes)
    return new_df

def linear_regression(x, y):
    model = LinearRegression()
    model.fit(x, y)
    b0 = model.intercept_
    b11, b12 = model.coef_

    return model, b0, b11, b12


to_predict = [[2567, 5], [1200, 2], [852, 5], [1852, 2], [1203, 3]]

data = pd.read_csv("datasets/house_price.csv")
no_outliers_data = remove_outliers(data)


x = []
y = np.array(no_outliers_data['price'].values)
for index, row in no_outliers_data.iterrows():
    x.append([int(row['area']), int(row['rooms'])])
x = (np.array(x)).reshape((-2, 2))

lin_regr_model, b0, b1_area, b1_rooms = linear_regression(x, y)

res = lin_regr_model.predict(to_predict)
for i in range(len(res)):
    print(f"For Area: {to_predict[i][0]} and Rooms: {to_predict[i][1]}")
    print(f"Predicted Price: {res[i]}")
    print("")


area_plane = np.linspace(0, 5000)
rooms_plane = np.linspace(0, 5)
area_plane, rooms_plane = np.meshgrid(area_plane, rooms_plane)
price_plane = area_plane*b1_area + rooms_plane*b1_rooms + b0


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(area_plane, rooms_plane, price_plane)
ax.scatter(no_outliers_data['area'].values, no_outliers_data['rooms'].values, no_outliers_data['price'].values, c=y, cmap='viridis', marker='o')
ax.set_xlabel("Area")
ax.set_ylabel("Rooms")
ax.set_zlabel("Price")
plt.show()