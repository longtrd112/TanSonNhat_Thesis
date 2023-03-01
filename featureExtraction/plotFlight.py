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
# flightDirectory = "../data/sortedData/2020-11-19/VJ245.csv"
# plotMAP(flightDirectory)
# plt.show()
