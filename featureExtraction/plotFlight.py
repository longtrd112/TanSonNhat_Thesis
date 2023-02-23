import os
from featureExtraction.utils.plotMAP import plotMAP, plt

dataDirectory = "../data/sortedData"  # Sorted flight data
for date in os.listdir(dataDirectory):
    dateDirectory = os.path.join(dataDirectory, date)

    for flight in os.listdir(dateDirectory):
        flightDirectory = os.path.join(dateDirectory, flight)
        plotMAP(flightDirectory)
        plt.show()

# Test specific flight

# flightDirectory = "../sortedData/2020-08-19/VN595.csv"
# plotMAP(flightDirectory)
# plt.show()
