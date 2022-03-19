import GetMatchStatistics

from fuzzywuzzy import process
from GetDataUrl import GetDataUrl

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

def target(FTHG, FTAG):
    if (FTHG < FTAG):
        return 1
    return 0
    
def GetAwayTeamWinProbability(leauge_id, home_team, away_team, year, month, 
                              B365CH, B365CD, B365CA, B365C_OVER, B365C_UNDER):
    
    df = GetDataUrl(leauge_id)
    df["target"] = np.vectorize(target)(df["FTHG"], df["FTAG"])
    
    team_data = pd.get_dummies(df[["HomeTeam", "AwayTeam"]])
    team_names = team_data.columns

    other = df[["Year", "Month", 
                "B365CH", "B365CD", "B365CA", "B365C>2.5", "B365C<2.5",
                "HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR"]]

    X = pd.concat([team_data, other], axis=1)
    y = df["target"]

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    model = LogisticRegression()
    model.fit(X, y)

    home_shots = GetMatchStatistics.GetHomeTeamShots(
        df=df, home_team=home_team, away_team=away_team, 
        B365CH=B365CH, B365CD=B365CD, B365CA=B365CA, B365C_OVER=B365C_OVER, B365C_UNDER=B365C_UNDER
    )

    away_shots = GetMatchStatistics.GetAwayTeamShots(
        df=df, home_team=home_team, away_team=away_team, 
        B365CH=B365CH, B365CD=B365CD, B365CA=B365CA, B365C_OVER=B365C_OVER, B365C_UNDER=B365C_UNDER
    )

    home_shots_target = GetMatchStatistics.GetHomeTeamShotsOnTarget(
        df=df, home_team=home_team, away_team=away_team, 
        B365CH=B365CH, B365CD=B365CD, B365CA=B365CA, B365C_OVER=B365C_OVER, B365C_UNDER=B365C_UNDER
    )

    away_shots_target = GetMatchStatistics.GetAwayTeamShotsOnTarget(
        df=df, home_team=home_team, away_team=away_team, 
        B365CH=B365CH, B365CD=B365CD, B365CA=B365CA, B365C_OVER=B365C_OVER, B365C_UNDER=B365C_UNDER
    )

    home_corners = GetMatchStatistics.GetHomeTeamCorners(
        df=df, home_team=home_team, away_team=away_team, 
        B365CH=B365CH, B365CD=B365CD, B365CA=B365CA, B365C_OVER=B365C_OVER, B365C_UNDER=B365C_UNDER
    )

    away_corners = GetMatchStatistics.GetAwayTeamCorners(
        df=df, home_team=home_team, away_team=away_team, 
        B365CH=B365CH, B365CD=B365CD, B365CA=B365CA, B365C_OVER=B365C_OVER, B365C_UNDER=B365C_UNDER
    )

    home_fouls = GetMatchStatistics.GetHomeTeamFoulsCommitted(
        df=df, home_team=home_team, away_team=away_team, 
        B365CH=B365CH, B365CD=B365CD, B365CA=B365CA, B365C_OVER=B365C_OVER, B365C_UNDER=B365C_UNDER
    )

    away_fouls = GetMatchStatistics.GetAwayTeamFoulsCommitted(
        df=df, home_team=home_team, away_team=away_team, 
        B365CH=B365CH, B365CD=B365CD, B365CA=B365CA, B365C_OVER=B365C_OVER, B365C_UNDER=B365C_UNDER
    )

    home_yellows = GetMatchStatistics.GetHomeTeamYellowCards(
        df=df, home_team=home_team, away_team=away_team, 
        B365CH=B365CH, B365CD=B365CD, B365CA=B365CA, B365C_OVER=B365C_OVER, B365C_UNDER=B365C_UNDER
    )

    away_yellows = GetMatchStatistics.GetAwayTeamYellowCards(
        df=df, home_team=home_team, away_team=away_team, 
        B365CH=B365CH, B365CD=B365CD, B365CA=B365CA, B365C_OVER=B365C_OVER, B365C_UNDER=B365C_UNDER
    )

    home_reds = GetMatchStatistics.GetHomeTeamRedCards(
        df=df, home_team=home_team, away_team=away_team, 
        B365CH=B365CH, B365CD=B365CD, B365CA=B365CA, B365C_OVER=B365C_OVER, B365C_UNDER=B365C_UNDER
    )

    away_reds = GetMatchStatistics.GetAwayTeamRedCards(
        df=df, home_team=home_team, away_team=away_team, 
        B365CH=B365CH, B365CD=B365CD, B365CA=B365CA, B365C_OVER=B365C_OVER, B365C_UNDER=B365C_UNDER
    )

    test = pd.DataFrame(columns=team_names, data=np.zeros(len(team_names)).reshape(1, len(team_names)))

    test[process.extractOne(f"HomeTeam_{home_team}", team_names)[0]] = 1
    test[process.extractOne(f"AwayTeam_{away_team}", team_names)[0]] = 1

    test["Year"] = year
    test["Month"] = month

    test["B365CH"] = B365CH
    test["B365CD"] = B365CD
    test["B365CA"] = B365CA
    test["B365C>2.5"] = B365C_OVER
    test["B365C<2.5"] = B365C_UNDER

    test["HS"] = home_shots
    test["AS"] = away_shots

    test["HST"] = home_shots_target
    test["AST"] = away_shots_target

    test["HC"] = home_corners
    test["AC"] = away_corners

    test["HF"] = home_fouls
    test["AF"] = away_fouls

    test["HY"] = home_yellows
    test["AY"] = away_yellows

    test["HR"] = home_reds
    test["AR"] = away_reds

    test = scaler.transform(test)

    return model.predict_proba(test)