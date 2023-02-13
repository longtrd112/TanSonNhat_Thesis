import os
import json
import pandas as pd
from datetime import date, timedelta, datetime
from adjusted_flight import Flight


def load_tma_config(file_name='tsn_arrival.json'):
    try:
        with open(os.path.join(os.path.abspath(__file__ + "/../"), file_name), "r") as f:
            tma_config = json.load(f)
        assert ('waypoint' and 'arrival_dict' and 'airport' and 'iaf' and 'entry_waypoint') in tma_config

        return tma_config
    except Exception:
        raise Exception(f"Problem with config file {file_name}.")


if __name__ == "__main__":
    start = datetime.now()

    # Load config of TMA
    config = load_tma_config()

    # Choosing starting and ending date
    start_date = date(2020, 5, 9)
    end_date = date(2021, 6, 24)
    df = pd.DataFrame()

    for date in (start_date + timedelta(n) for n in range(int((end_date - start_date).days) + 1)):
        folder_path = os.path.join(os.path.abspath(__file__ + "/../../sortedData"), date.strftime("%Y-%m-%d"))

        if not os.path.exists(folder_path):
            continue

        for filename in os.listdir(folder_path):

            file_path = os.path.join(folder_path, filename)

            try:
                flight = Flight(file_path, config)

                data_dict = {"flight": filename.split(".")[0], "date": date,
                             "entry_waypoint": flight.entry_waypoint,
                             "landing_runway": flight.landing_runway,
                             "entry_latitude": flight.traj[0][0], "entry_longitude": flight.traj[0][1],
                             "entry_altitude": flight.traj[0][4], "entry_ground_speed": flight.traj[0][2],
                             "entry_time": flight.traj[0][3], "arrival_time": flight.traj[flight.landing_data][3],
                             "entry_time_HCM": flight.entry_time_HCM, "arrival_time_HCM": flight.arrival_time_HCM}

                df_dictionary = pd.DataFrame([data_dict])
                df = pd.concat([df, df_dictionary], ignore_index=True)

            except Exception as e:
                print(filename, e)
                continue

    df.to_csv(os.path.join(os.path.abspath(__file__ + '/../'), "extracted_features.csv"), index=False)
    print(datetime.now() - start)
