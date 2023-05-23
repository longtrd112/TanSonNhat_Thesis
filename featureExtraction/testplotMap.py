import matplotlib.pyplot as plt
import numpy as np

# create some sample data
latitude = np.array([51.5074, 40.7128, 19.4326, -33.8688])
longitude = np.array([-0.1278, -74.0060, -99.1332, 151.2093])
magnitude = np.array([10, 5, 2, 8])

# plot the data
fig, ax = plt.subplots()
scatter = ax.scatter(longitude, latitude, c=magnitude)

# add a colorbar
plt.colorbar(scatter)

# set the x and y axis labels
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

# set the title
ax.set_title('Magnitude of data points')

# show the plot
plt.show()
