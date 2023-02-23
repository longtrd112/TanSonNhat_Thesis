import pandas as pd
import datetime
import pytz
import os
import shutil
import geopy.distance as gpd

# Copy original data to working directory
dataDirectory = "../data"
extracted_data_directory = "../data/Extract"
sorted_data_directory = "../data/sortedData"

if os.path.exists(sorted_data_directory):
    shutil.rmtree(sorted_data_directory)
    os.mkdir(sorted_data_directory)

else:
    os.mkdir(sorted_data_directory)

for date in os.listdir(extracted_data_directory):
    date_directory_extracted = os.path.join(extracted_data_directory, date)
    date_directory_sorted = os.path.join(sorted_data_directory, date)

    shutil.copytree(date_directory_extracted, date_directory_sorted)


# Timestamp converting and sorting
for date in os.listdir(sorted_data_directory):
    dateDirectory = os.path.join(sorted_data_directory, date)

    for flight in os.listdir(dateDirectory):
        flightDirectory = os.path.join(dateDirectory, flight)
        dataFile = pd.read_csv(flightDirectory)

        flight_date = []
        flight_time = []
        epochTime = dataFile.timestamp.to_list()

        # Convert to GMT+7 time zone
        for time in epochTime:
            timeZoneHCM = pytz.timezone("Asia/Ho_Chi_Minh")
            date = datetime.datetime.fromtimestamp(time, timeZoneHCM).date()
            time = datetime.datetime.fromtimestamp(time, timeZoneHCM).time()
            flight_date.append(date)
            flight_time.append(time)

        dataFile['flight_date'] = flight_date
        dataFile['flight_time'] = flight_time

        # Drop out duplicate data points
        dropDuplicatesFile = dataFile.drop_duplicates(subset=['timestamp'])

        # Sort by time
        sortedFile = dropDuplicatesFile.sort_values(by=['timestamp'], ascending=True)
        sortedFile.to_csv(flightDirectory, index=False)

# Delete flight with extremely low number of data points
for date in os.listdir(sorted_data_directory):
    dateDirectory = os.path.join(sorted_data_directory, date)

    for flight in os.listdir(dateDirectory):
        flightDirectory = os.path.join(dateDirectory, flight)
        dataFile = pd.read_csv(flightDirectory)

        # 50 data points ~ Data within 50 minutes
        if len(dataFile) < 50:
            os.remove(flightDirectory)


# Detecting flight with errors
def find_final_location(df):
    lastLongitude = df['longitude'].iloc[-1]
    lastLatitude = df['latitude'].iloc[-1]
    return [lastLatitude, lastLongitude]


def append_error(d, f, e, date, flight, error):
    d.append(date)
    f.append(flight)
    e.append(error)


def detect_multiple_flight(df):
    for i in range(len(df) - 1):
        # TIme difference between 2 data points > 3600s --> Landing - take-off
        if df['timestamp'].iloc[i + 1] - df['timestamp'].iloc[i] > 3600:
            return True
    return False


def distance(a, b):
    # [] = [latitude, longitude,...]
    first_point_coord = [a[0], a[1]]
    second_point_coord = [b[0], b[1]]

    return gpd.geodesic(first_point_coord, second_point_coord).km


d = []  # List of date having error flights
f = []  # List of error flights
e = []  # List of errors

airport = [10.8188, 106.652]

for date in os.listdir(sorted_data_directory):
    dateDirectory = os.path.join(sorted_data_directory, date)

    for flight in os.listdir(dateDirectory):
        flightDirectory = os.path.join(dateDirectory, flight)
        dataFile = pd.read_csv(flightDirectory)

        finalLocation = find_final_location(dataFile)
        distanceFromAirport = distance(finalLocation, airport)

        if dataFile['altitude'].iloc[-1] > 500:
            append_error(d, f, e, date, flight, "Did not land.")

        elif distanceFromAirport > 20:  # Distance(airport, iaf) ~ 18 km
            append_error(d, f, e, date, flight, "Did not land at TSN.")

        elif detect_multiple_flight(dataFile):
            append_error(d, f, e, date, flight, "Multiple flights detected.")

flightError = {'date': d, 'flight': f, 'error': e}
flightErrorCSV = pd.DataFrame(data=flightError)
flightErrorCSV.to_csv("../data/flightError.csv", index=False)

# Delete flight containing errors
errorData = pd.read_csv("../data/flightError.csv")
for i in range(0, len(errorData)):
    date = errorData['date'].iloc[i]
    flight = errorData['flight'].iloc[i]

    delete_file_directory = os.path.join(sorted_data_directory, date, flight)
    os.remove(delete_file_directory)
