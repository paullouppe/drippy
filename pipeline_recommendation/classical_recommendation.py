import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
from scipy.spatial.distance import cosine
import ollama

def tfidf(df: pd.DataFrame):

    transformer = TfidfTransformer()

    return transformer.fit_transform(df)

def embeded_similarity(df: pd.DataFrame, model: str):

    embeded_matrix = []

    for idx, row in df.iterrows():
        row_str = "; ".join([f"{col}: {row[col]}" for col in row.index])
        embeded_matrix.append(ollama.embed(model = model, input = row_str))

    return np.array(embeded_matrix)

def run_recommendation_pipeline(df: pd.DataFrame, reco_algo: function, random_state: int = None, users_data: pd.DataFrame = None, extra_ra_args = []):
    
    np.random.seed(random_state)

    # Run given algorithm on data to create matrix
    encoded_matrix = reco_algo(df.values(), **extra_ra_args)

    # TODO User profiles (pas tip top là)
    users_profiles = []

    if not users_data: 
        users_profiles = [np.random().rand(1, len(df.columns))] # TODO Pas top ; pas de négatif ou quoi, juste entre 0 et 1 etc.
    if users_data:
        # We should be able to generate profiles with real datas
        users_profiles = reco_algo(users_data.values(), **extra_ra_args) # Just that ? Not sure

    # Compare the two with cosine similarity
    similarities = []

    for user in users_profiles.to_numpy():

        user_similarities = []

        for outfit in encoded_matrix:
            user_similarities.append(cosine(user, outfit))

        similarities.append(user_similarities)

    return similarities