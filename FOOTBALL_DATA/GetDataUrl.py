import numpy as np
import pandas as pd

def GetDataUrl(leauge_id):
    
    data = [
        f"https://www.football-data.co.uk/mmz4281/1920/{leauge_id}.csv",
        f"https://www.football-data.co.uk/mmz4281/2021/{leauge_id}.csv",
        f"https://www.football-data.co.uk/mmz4281/2122/{leauge_id}.csv"
    ]

    li = []

    for season in data:
        df = pd.read_csv(season)
        li.append(df)

    df = pd.concat(li)

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df["Year"] = df["Date"].apply(lambda date: date.year)
    df["Month"] = df["Date"].apply(lambda date: date.month)

    df = df[[
        "Year", "Month", "HomeTeam", "AwayTeam", "FTHG", "FTAG", 
        "B365CH", "B365CD", "B365CA", "B365C>2.5", "B365C<2.5", 
        "HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR"
    ]]

    if 0 == df.isnull().sum().sum():
        return df
    else:
        print(df.isnull().sum().sum())
        df = df.dropna()
        return df