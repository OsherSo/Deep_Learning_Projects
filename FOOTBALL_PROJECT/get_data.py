import numpy as np
import pandas as pd
from data import leagues_data


def target(home_goals, away_goals):
    if home_goals > away_goals:
        return 1
    return 0


def get_data(league_name):
    data = leagues_data[league_name]
    li = []
    for season in data:
        df = pd.read_csv(season)
        li.append(df)

    df = pd.concat(li)
    df = df[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "B365CH", "B365CD", "B365CA", "B365C>2.5", "B365C<2.5"]]
    print(f"number of null: {df.isnull().sum().sum()}\n")
    df = df.dropna()

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df = df.sort_values("Date")

    df["target"] = np.vectorize(target)(df["FTHG"], df["FTAG"])

    df = df.reset_index(drop=True)
    return df
