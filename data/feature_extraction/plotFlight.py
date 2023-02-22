import os
from data.utils.plotMAP import plotMAP, plt

# # dataDirectory = "../Extract"  # Original flight data
dataDirectory = "../sortedData"  # Sorted flight data
for day in os.listdir(dataDirectory):

    dayDirectory = os.path.join(dataDirectory, day)

    for flight in os.listdir(dayDirectory):

        flightDirectory = os.path.join(dayDirectory, flight)

        plotMAP(flightDirectory)

        plt.show()


# Test specific flight

# flightDirectory = "../sortedData/2020-08-19/VN595.csv"
# plotMAP(flightDirectory)
# plt.show()
