import pandas as pd
import os

sortedDataDirectory = "../data/sortedData"
errorData = pd.read_csv("../data/feature_extraction/flightError.csv")

for i in range(0, len(errorData)):
    day = errorData['day'].iloc[i]
    flight = errorData['flight'].iloc[i]

    deleteFileDirectory = os.path.join(sortedDataDirectory, day, flight)
    os.remove(deleteFileDirectory)
