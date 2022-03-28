import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from get_data import get_data


def prepare_data(league_name):
    df = get_data(league_name)

    team_data = pd.get_dummies(df[["HomeTeam", "AwayTeam"]])
    team_columns = team_data.columns

    odds = df[["B365CH", "B365CD", "B365CA", "B365C>2.5", "B365C<2.5"]]

    x = pd.concat([team_data, odds], axis=1)
    y = df["target"]

    test_size_1 = int(0.7 * len(x))

    x_train = x[:test_size_1]
    y_train = y[:test_size_1]

    x_other = x[test_size_1:]
    y_other = y[test_size_1:]

    test_size_2 = int(0.5 * len(x_other))

    x_val = x_other[:test_size_2]
    y_val = y_other[:test_size_2]

    x_test = x_other[test_size_2:]
    y_test = y_other[test_size_2:]

    scaler = MinMaxScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    print(f"Train : {x_train.shape}, {y_train.shape}")
    print(f"Validation : {x_val.shape}, {y_val.shape}")
    print(f"Test : {x_test.shape}, {y_test.shape}\n")

    return x_train, y_train, x_val, y_val, x_test, y_test, scaler, team_columns
