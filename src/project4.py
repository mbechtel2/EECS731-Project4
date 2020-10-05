################################################################################
#
# File: project4
# Author: Michael Bechtel
# Date: October 5, 2020
# Class: EECS 731
# Description: Use regression models to predict the number of points scored
#               by a NBA team based on features from previous games (points scored, etc.)
# 
################################################################################

# Python imports
import pandas as pd
pd.set_option("display.precision",2)
from math import sqrt

from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# Create Regression models
linearReg_model = linear_model.LinearRegression()
randomForest_model = RandomForestClassifier(max_depth=6)
gradBoost_model = GradientBoostingClassifier()
neuralNetwork_model = MLPRegressor(max_iter=10000)

# Read the raw dataset
game_list = pd.read_csv("../data/raw/nbaallelo.csv")

# Create list of team abbreviations for current NBA teams
team_codes = ["ATL","BRK","BOS","CHA","CHI","CLE","DAL","DEN","DET","GSW","HOU",
              "IND","LAC","LAL","MEM","MIA","MIL","MIN","NOP","NYK","OKC","ORL",
              "PHI","PHO","POR","SAC","SAS","TOR","UTA","WAS"]
              
# Create lists for holding data from final dataframe         
team_num_games = [] # Number of games used for regression
linearReg_results = []
randomForest_results = []
gradBoost_results = []
neuralNetwork_results = []

# Peform Regression for each NBA team
print("\nPerforming Regression for:")
for team in team_codes:
    # Get the games that the current team played against other current NBA franchises
    team_list = game_list[game_list["team_id"].isin([team])]
    team_list = team_list[team_list["opp_id"].isin(team_codes)]
    
    # Retrieve the desired columns, drop any rows with empty values and save the dataset
    team_list = team_list.loc[:,("team_id","pts","win_equiv","opp_id","opp_pts","game_location","forecast")].dropna()
    team_list.to_csv("../data/processed/{}.csv".format(team))
    
    # Store the number of games in the dataset
    team_num_games.append(len(team_list))
    
    # Print the current team and number of games
    #   Used to track overall progress
    print("{} ({} games)".format(team,len(team_list)))

    # Separate the dataset into feature and point datasets
    team_features = team_list.loc[:,("team_id","win_equiv","opp_id","opp_pts","game_location","forecast")].dropna()
    team_points = team_list.loc[:,"pts"].dropna()
    
    # Perform one-hot encoding where necessary
    team_features = pd.get_dummies(team_features, columns=["team_id","opp_id","game_location"])

    # Create training and testing sets
    features_train, features_test, points_train, points_test = train_test_split(team_features, team_points, test_size=0.50, random_state=0, shuffle=True)

    # Perform Linear Regression
    points_pred = linearReg_model.fit(features_train, points_train).predict(features_test)
    linearReg_results.append(sqrt(mean_squared_error(points_test, points_pred)))

    # Perform Random Forest
    points_pred = randomForest_model.fit(features_train, points_train).predict(features_test)
    randomForest_results.append(sqrt(mean_squared_error(points_test, points_pred)))

    # Perform Gradient Boosting
    points_pred = gradBoost_model.fit(features_train, points_train).predict(features_test)
    gradBoost_results.append(sqrt(mean_squared_error(points_test, points_pred)))

    # Perform Neural Network
    points_pred = neuralNetwork_model.fit(features_train, points_train).predict(features_test)
    neuralNetwork_results.append(sqrt(mean_squared_error(points_test, points_pred)))
    
    print("\tDone")
    
# Create a new dataframe with the results and display it
print()
print("===========================================")
print("Results")
print("===========================================")

reg_results = pd.DataFrame({"Team":team_codes,"# of Games":team_num_games,
                            "LinearRegression":linearReg_results,"RandomForest":randomForest_results,
                            "GradientBoost":gradBoost_results,"NeuralNetwork":neuralNetwork_results})
print(reg_results)                         