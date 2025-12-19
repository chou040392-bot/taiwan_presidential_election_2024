import sqlite3
import pandas as pd
import numpy as np

connection  = sqlite3.connect("data/taiwan_presidential_election_2024.db")
votes_by_village  = pd.read_sql("""SELECT * FROM votes_by_village;""", con = connection)
connection.close()
total_votes = votes_by_village["sum_votes"].sum()
votes_by_village.groupby("number")["sum_votes"].sum()
country_percentage = votes_by_village.groupby("number")["sum_votes"].sum() / total_votes

vector_a = country_percentage.values
print(vector_a)

groupby_variables = ["county", "town", "village"]
village_total_votes = votes_by_village.groupby(groupby_variables)["sum_votes"].sum().reset_index()
print(village_total_votes)
merged = pd.merge(votes_by_village, village_total_votes, left_on=groupby_variables, right_on=groupby_variables,
                  how="left")
merged["village_percentage"] = merged["sum_votes_x"] / merged["sum_votes_y"]
merged = merged[["county", "town", "village", "number", "village_percentage"]]
pivot_df = merged.pivot(index=["county", "town", "village"], columns="number", values="village_percentage").reset_index()
pivot_df = pivot_df.rename_axis(None, axis=1)

cosine_similarities = []
for row in pivot_df.iterrows():
    vector_bi = np.array([row[1][1], row[1][2], row[1][3]])
    vector_a_dot_vector_bi = np.dot(vector_a, vector_bi)
    length_vector_a = pow((vector_a**2).sum(), 0.5)
    length_vector_bi = pow((vector_bi**2).sum(), 0.5)
    cosine_similarity = vector_a_dot_vector_bi / (length_vector_a*length_vector_bi)
    cosine_similarities.append(cosine_similarity)
print(cosine_similarities)