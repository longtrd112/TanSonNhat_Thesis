import pandas as pd
import datetime
import pytz
import os

dataDirectory = "../data/Extract"
for day in os.listdir(dataDirectory):

    dayDirectory = os.path.join(dataDirectory, day)

    for flight in os.listdir(dayDirectory):

        flightDirectory = os.path.join(dayDirectory, flight)
        dataFile = pd.read_csv(flightDirectory)

        flight_date = []
        flight_time = []
        epochTime = dataFile.timestamp.to_list()

        for time in epochTime:
            timeZoneHCM = pytz.timezone("Asia/Ho_Chi_Minh")
            date = datetime.datetime.fromtimestamp(time, timeZoneHCM).date()
            time = datetime.datetime.fromtimestamp(time, timeZoneHCM).time()
            flight_date.append(date)
            flight_time.append(time)

        dataFile['flight_date'] = flight_date
        dataFile['flight_time'] = flight_time

        # Drop out duplicate data
        dropDuplicatesFile = dataFile.drop_duplicates(subset=['timestamp'])

        # Sort by time
        sortedFile = dropDuplicatesFile.sort_values(by=['timestamp'], ascending=True)
        sortedFile.to_csv(flightDirectory)
