# Maitar Asher
# ma4265
import numpy as np
import matplotlib.pyplot as plt

D = np.array([[0, 206, 429, 1504, 963, 2976, 3095, 2979, 1949],
              [206, 0, 233, 1308, 802, 2815, 2934, 2786, 1771],
              [429, 233, 0, 1075, 671, 2684, 2799, 2631, 1616],
              [1504, 1308, 1075, 0, 1329, 3273, 3053, 2687, 2037],
              [963, 802, 671, 1329, 0, 2013, 2142, 2054, 996],
              [2976, 2815, 2684, 3273, 2013, 0, 808, 1131, 1307],
              [3095, 2934, 2799, 3053, 2142, 808, 0, 379, 1235],
              [2979, 2786, 2631, 2687, 2054, 1131, 379, 0, 1059],
              [1949, 1771, 1616, 2037, 996, 1307, 1235, 1059, 0]])

d = D.shape[0]
"""
Initialize xi=[lat , lon] values for cities randomly  (input to np.random.rand() is the size of array)
Latitude and longitude are a pair of numbers (coordinates) used to describe a position on the plane of a geographic coordinate system. 
The numbers are in decimal degrees format and range from -90 to 90 for latitude and -180 to 180 for longitude.
"""

# Generate random latitude and longitude values between -90 to 90 and -180 to 180 respectively
latitudes = np.random.uniform(low=-90.0, high=90.0, size=(9,))
longitudes = np.random.uniform(low=-180.0, high=180.0, size=(9,))

# Combine the latitude and longitude values to create the final array of coordinates
coordinates = np.column_stack((latitudes, longitudes))
print(coordinates)

"""Embedding discrepancy function"""
def discrepancy(X):
    disr = 0
    for i in range(d):
        for j in range(d):
            disr += (np.linalg.norm(X[i,:] - X[j,:]) - D[i][j]) ** 2
    return disr

print(discrepancy(coordinates))

def dis_gradient(X):
    gradient = np.zeros((d, 2))
    for i in range(d):
        for j in range(d):
            if i != j:
                dist = X[i, :] - X[j, :]
                gradient[i, :] += (4 * (np.linalg.norm(dist) - D[i][j]) * (dist))/ np.linalg.norm(dist)
    return gradient


learning_rate = 0.001

"""
Gradient descent optimization
"""
for i in range(10000):
    gradient = dis_gradient(coordinates)
    coordinates -= learning_rate * gradient

cities = ['BOS', 'NYC', 'DC', 'MIA', 'CHI', 'SEA', 'SF', 'LA', 'DEN']
"""
Estimated locations:
"""
for i in range(d):
    print(cities[i],": lat-", coordinates[i][0], "lon-", coordinates[i][1])

fig, ax = plt.subplots()
ax.scatter(coordinates[:, 0], coordinates[:, 1])
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Estimated Locations of Cities')
for i in range(d):
    ax.annotate(cities[i], (coordinates[i][0], coordinates[i][1]), textcoords="offset points", xytext=(5,5), ha='center')
plt.show()