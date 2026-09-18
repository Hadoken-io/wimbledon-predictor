import pandas as pd
df = pd.read_csv(r"data_processed\features.csv")
grass = df[df["Surf_Grass"]==1]
recent = grass[grass["Date"] >= "2024-01-01"]
all_players = pd.concat([recent["Player_1"], recent["Player_2"]]).value_counts()
print(all_players.head(30))