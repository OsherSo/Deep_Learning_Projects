from fuzzywuzzy import process

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

def RunModel(df, y, home_team, away_team, B365CH, B365CD, B365CA, B365C_OVER, B365C_UNDER):
    team_data = pd.get_dummies(df[["HomeTeam", "AwayTeam"]])
    team_names = team_data.columns
    
    odds = df[["B365CH","B365CD","B365CA","B365C>2.5","B365C<2.5"]]
    
    X = pd.concat([team_data, odds], axis=1)
    
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X, y)

    test = pd.DataFrame(columns=team_names, data=np.zeros(len(team_names)).reshape(1, len(team_names)))
        
    test[process.extractOne(f"HomeTeam_{home_team}", team_names)[0]] = 1
    test[process.extractOne(f"AwayTeam_{away_team}", team_names)[0]] = 1
    
    test["B365CH"] = B365CH
    test["B365CD"] = B365CD
    test["B365CA"] = B365CA
    test["B365C>2.5"] = B365C_OVER
    test["B365C<2.5"] = B365C_UNDER
    
    test = scaler.transform(test)
        
    return model.predict(test)[0]

    
def GetHomeTeamShots(df, home_team, away_team, B365CH, B365CD, B365CA, B365C_OVER, B365C_UNDER):
    if (0 == df.isnull().sum().sum()):
        y = df["HS"] # Home Team Shots.
        return RunModel(df=df, y=y, home_team=home_team, away_team=away_team, 
                        B365CH=B365CH, B365CD=B365CD, B365CA=B365CA, 
                        B365C_OVER=B365C_OVER, B365C_UNDER=B365C_UNDER)
    else:
        return -1

def GetAwayTeamShots(df, home_team, away_team, B365CH, B365CD, B365CA, B365C_OVER, B365C_UNDER):
    if (0 == df.isnull().sum().sum()):
        y = df["AS"] # Away Team Shots.
        return RunModel(df=df, y=y, home_team=home_team, away_team=away_team, 
                        B365CH=B365CH, B365CD=B365CD, B365CA=B365CA, 
                        B365C_OVER=B365C_OVER, B365C_UNDER=B365C_UNDER)
    else:
        return -1

def GetHomeTeamShotsOnTarget(df, home_team, away_team, B365CH, B365CD, B365CA, B365C_OVER, B365C_UNDER):
    if (0 == df.isnull().sum().sum()):
        y = df["HST"] # Home Team Shots on Target.
        return RunModel(df=df, y=y, home_team=home_team, away_team=away_team, 
                        B365CH=B365CH, B365CD=B365CD, B365CA=B365CA, 
                        B365C_OVER=B365C_OVER, B365C_UNDER=B365C_UNDER)
    else:
        return -1

def GetAwayTeamShotsOnTarget(df, home_team, away_team, B365CH, B365CD, B365CA, B365C_OVER, B365C_UNDER):
    if (0 == df.isnull().sum().sum()):
        y = df["AST"] # Away Team Shots on Target.
        return RunModel(df=df, y=y, home_team=home_team, away_team=away_team, 
                        B365CH=B365CH, B365CD=B365CD, B365CA=B365CA, 
                        B365C_OVER=B365C_OVER, B365C_UNDER=B365C_UNDER)
    else:
        return -1

def GetHomeTeamCorners(df, home_team, away_team, B365CH, B365CD, B365CA, B365C_OVER, B365C_UNDER):
    if (0 == df.isnull().sum().sum()):
        y = df["HC"] # Home Team Corners.
        return RunModel(df=df, y=y, home_team=home_team, away_team=away_team, 
                        B365CH=B365CH, B365CD=B365CD, B365CA=B365CA, 
                        B365C_OVER=B365C_OVER, B365C_UNDER=B365C_UNDER)
    else:
        return -1

def GetAwayTeamCorners(df, home_team, away_team, B365CH, B365CD, B365CA, B365C_OVER, B365C_UNDER):
    if (0 == df.isnull().sum().sum()):
        y = df["AC"] # Away Team Corners.
        return RunModel(df=df, y=y, home_team=home_team, away_team=away_team, 
                        B365CH=B365CH, B365CD=B365CD, B365CA=B365CA, 
                        B365C_OVER=B365C_OVER, B365C_UNDER=B365C_UNDER)
    else:
        return -1

def GetHomeTeamFoulsCommitted(df, home_team, away_team, B365CH, B365CD, B365CA, B365C_OVER, B365C_UNDER):
    if (0 == df.isnull().sum().sum()):
        y = df["HF"] # Home Team Fouls Committed.
        return RunModel(df=df, y=y, home_team=home_team, away_team=away_team, 
                        B365CH=B365CH, B365CD=B365CD, B365CA=B365CA, 
                        B365C_OVER=B365C_OVER, B365C_UNDER=B365C_UNDER)
    else:
        return -1

def GetAwayTeamFoulsCommitted(df, home_team, away_team, B365CH, B365CD, B365CA, B365C_OVER, B365C_UNDER):
    if (0 == df.isnull().sum().sum()):
        y = df["AF"] # Away Team Fouls Committed.
        return RunModel(df=df, y=y, home_team=home_team, away_team=away_team, 
                        B365CH=B365CH, B365CD=B365CD, B365CA=B365CA, 
                        B365C_OVER=B365C_OVER, B365C_UNDER=B365C_UNDER)
    else:
        return -1

def GetHomeTeamYellowCards(df, home_team, away_team, B365CH, B365CD, B365CA, B365C_OVER, B365C_UNDER):
    if (0 == df.isnull().sum().sum()):
        y = df["HY"] # Home Team Yellow Cards.
        return RunModel(df=df, y=y, home_team=home_team, away_team=away_team, 
                        B365CH=B365CH, B365CD=B365CD, B365CA=B365CA, 
                        B365C_OVER=B365C_OVER, B365C_UNDER=B365C_UNDER)
    else:
        return -1

def GetAwayTeamYellowCards(df, home_team, away_team, B365CH, B365CD, B365CA, B365C_OVER, B365C_UNDER):
    if (0 == df.isnull().sum().sum()):
        y = df["AY"] # Away Team Yellow Cards.
        return RunModel(df=df, y=y, home_team=home_team, away_team=away_team, 
                        B365CH=B365CH, B365CD=B365CD, B365CA=B365CA, 
                        B365C_OVER=B365C_OVER, B365C_UNDER=B365C_UNDER)
    else:
        return -1

def GetHomeTeamRedCards(df, home_team, away_team, B365CH, B365CD, B365CA, B365C_OVER, B365C_UNDER):
    if (0 == df.isnull().sum().sum()):
        y = df["HR"] # Home Team Red Cards.
        return RunModel(df=df, y=y, home_team=home_team, away_team=away_team, 
                        B365CH=B365CH, B365CD=B365CD, B365CA=B365CA, 
                        B365C_OVER=B365C_OVER, B365C_UNDER=B365C_UNDER)
    else:
        return -1

def GetAwayTeamRedCards(df, home_team, away_team, B365CH, B365CD, B365CA, B365C_OVER, B365C_UNDER):
    if (0 == df.isnull().sum().sum()):
        y = df["AR"] # Away Team Red Cards.
        return RunModel(df=df, y=y, home_team=home_team, away_team=away_team, 
                        B365CH=B365CH, B365CD=B365CD, B365CA=B365CA, 
                        B365C_OVER=B365C_OVER, B365C_UNDER=B365C_UNDER)
    else:
        return -1
