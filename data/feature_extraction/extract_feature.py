from datetime import date, timedelta, datetime
import os
import pandas as pd
import json
from flight import Flight


def load_tma_config(file_name='tsn_arrival.json'):
    try:
        with open(os.path.join(os.path.abspath(__file__ + "/../"), file_name), "r") as f:
            tma_config = json.load(f)
        assert ('waypoint' and 'arrival_dict' and 'arrival_dict_v' and 'holding_waypoints'
                and 'airport' and 'iaf' and 'confusing') in tma_config
        return tma_config
    except Exception:
        raise Exception(f"Problem with config file {file_name}.")


if __name__ == "__main__":
    start = datetime.now()

    # Load config of TMA
    config = load_tma_config()

    # Choosing starting and ending date
    start_date = date(2020, 5, 9)
    end_date = date(2020, 5, 9)
    df = pd.DataFrame()

    for date in (start_date + timedelta(n) for n in range(int((end_date - start_date).days) + 1)):
        folder_path = os.path.join(os.path.abspath(__file__ + "/../../Test"), date.strftime("%Y-%m-%d"))

        if not os.path.exists(folder_path):
            continue

        for filename in os.listdir(folder_path):

            file_path = os.path.join(folder_path, filename)

            try:
                flight = Flight(file_path, config)
                if True:  # flight.coming_to_LIMES() ?
                    flight.desegment()
                    data_dict = {"flight": filename.split(".")[0], "date": date, "route": flight.arrival,
                                 'entry_time': flight.segments[list(flight.segments)[0]][0][3],
                                 'arrival_time': flight.segments[list(flight.segments)[-1]][-1][3]}

                    for wp in flight.segments:

                        data_dict[f'v_{wp}'] = flight.segments[wp][0][2]  # Ground speed
                        data_dict[f'a_{wp}'] = flight.segments[wp][0][4]  # Altitude

                        holding_time = 0
                        for i in range(len(flight.segments[wp]) - 1):
                            if (flight.segments[wp][i][0:2] == flight.segments[wp][i + 1][0:2]).all():
                                holding_time += flight.segments[wp][i + 1][3] - flight.segments[wp][i][3]

                        data_dict[f't_{wp}'] = flight.segments[wp][-1][3] - flight.segments[wp][0][3] - holding_time
                        data_dict[f'{wp}'] = flight.segments[wp][0][3]

                    # Include holding information
                    holding = flight.holding
                    for h in holding:
                        if f'h_{h[0]}' not in data_dict:
                            data_dict[f'h_{h[0]}'] = h[1]
                        else:
                            data_dict[f'h_{h[0]}'] += h[1]

                    # appearance_info = flight.get_info_in_advance(50, data_dict['entry_time'])
                    # data_dict['appear_time'] = appearance_info[0]
                    # data_dict['appear_lat'] = appearance_info[1]
                    # data_dict['appear_lon'] = appearance_info[2]
                    # data_dict['appear_alt'] = appearance_info[3]
                    # data_dict['appear_heading'] = appearance_info[4]
                    # data_dict['appear_speed'] = appearance_info[5]

                    df_dictionary = pd.DataFrame([data_dict])
                    df = pd.concat([df, df_dictionary], ignore_index=True)

            except Exception as e:
                print(filename, e)
                continue

    df.to_csv(os.path.join(os.path.abspath(__file__ + '/../'), "test_extracted_data.csv"), index=False)
    print(datetime.now() - start)
