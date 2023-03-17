import os
import json
import pandas as pd
from datetime import date, timedelta, datetime
from flightFeatures import Flight


def load_airport_config(file_name='tsn_arrival.json'):
    try:
        with open(os.path.join(os.path.abspath(__file__ + "/../"), file_name), "r") as f:
            tma_config = json.load(f)

        return tma_config

    except Exception:
        raise Exception(f"Problem with config file {file_name}.")


if __name__ == "__main__":
    start = datetime.now()

    # Load config of TMA
    config = load_airport_config()

    # Choosing starting and ending date
    start_date = date(2020, 5, 9)
    end_date = date(2021, 4, 27)
    df = pd.DataFrame()

    for date in (start_date + timedelta(n) for n in range(int((end_date - start_date).days) + 1)):
        print(date)
        dateDirectory = os.path.join(os.path.abspath(__file__ + "/../../data/sortedData"), date.strftime("%Y-%m-%d"))

        if not os.path.exists(dateDirectory):
            continue

        for flight_name in os.listdir(dateDirectory):
            flightDirectory = os.path.join(dateDirectory, flight_name)

            try:
                flight = Flight(flightDirectory, config)

                data_dict = {"flight": flight_name.split(".")[0], "date": date,
                             "landing_runway": flight.landing_runway,

                             "first_latitude": flight.first_point_data[0] - 10.8188,
                             "first_longitude": flight.first_point_data[1] - 106.652,
                             "first_altitude": flight.first_point_data[4],
                             "first_ground_speed": flight.first_point_data[2],
                             "first_heading_angle": flight.first_point_data[5],

                             "second_latitude": flight.second_point_data[0] - 10.8188,
                             "second_longitude": flight.second_point_data[1] - 106.652,
                             "second_altitude": flight.second_point_data[4],
                             "second_ground_speed": flight.second_point_data[2],
                             "second_heading_angle": flight.second_point_data[5],


                             "entry_latitude": flight.traj[0][0] - 10.8188,
                             "entry_longitude": flight.traj[0][1] - 106.652,
                             "entry_altitude": flight.traj[0][4], "entry_ground_speed": flight.traj[0][2],
                             "entry_heading_angle": flight.traj[0][5],

                             "entry_time": flight.traj[0][3], "arrival_time": flight.traj[flight.landing_data][3],
                             "transit_time": flight.traj[flight.landing_data][3] - flight.traj[0][3],
                             "distance_to_airport": flight.distance_to_airport, "model_type": flight.type}

                df_dictionary = pd.DataFrame([data_dict])
                df = pd.concat([df, df_dictionary], ignore_index=True)

            except Exception as e:
                print(flight_name, e)
                continue

    df.to_csv(os.path.join(os.path.abspath(__file__ + '/../'), "extracted_features_3points.csv"), index=False)

    print("Execution time: ", datetime.now() - start)
