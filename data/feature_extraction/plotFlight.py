import os
from data.utils.plotMAP import plotMAP, plt

# dataDirectory = "../Extract"  # Original flight data
dataDirectory = "../Test"  # Sorted flight data
for day in os.listdir(dataDirectory):

    dayDirectory = os.path.join(dataDirectory, day)

    for flight in os.listdir(dayDirectory):

        flightDirectory = os.path.join(dayDirectory, flight)

        plotMAP(flightDirectory)

        plt.show()


# Test specific flight

# flightDirectory = "../sortedData/2020-05-11/VJ326.csv"
# plotMAP(flightDirectory)
# plt.show()
