import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
from scipy.spatial.distance import cosine
import ollama
from tqdm import tqdm
import random

def tfidf(df: pd.DataFrame):

    transformer = TfidfTransformer()

    return transformer.fit_transform(df.values).toarray()

def embeded_similarity(df: pd.DataFrame, model: str):

    embeded_matrix = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Embedding with {model}"):
        row_str = "; ".join([f"{col}: {row[col]}" for col in row.index])
        embeded_matrix.append(ollama.embed(model = model, input = row_str)["embeddings"][0])

    return np.array(embeded_matrix)

def run_recommendation_pipeline(df: pd.DataFrame, reco_algo, users_liked: list, extra_ra_args = {}):
    
    # Run given algorithm on data to create matrix
    encoded_matrix = reco_algo(df, **extra_ra_args)

    data_nbr = len(encoded_matrix[0])

    # User profiles
    users_profiles = []

    for likes in users_liked:
        
        profile = np.zeros(data_nbr)

        for i in range(data_nbr):
            
            score = 0
            nb_app = 0

            for like_id in likes:
                
                score += encoded_matrix[like_id][i]
                nb_app += 1

            profile[i] = score / nb_app

        users_profiles.append(profile)

    # Compare the two with cosine similarity
    similarities = []

    for user in users_profiles:

        user_similarities = []

        for outfit in encoded_matrix:
            user_similarities.append(cosine(user, outfit))

        similarities.append(user_similarities)

    return similarities

def try_from_pickle(path: str, reco_algo, extra_ra_args = {}):

    rd = random.Random()

    # Get the pickled data and one-hot encode it
    clothes = pd.DataFrame(pd.read_pickle(path))
    clothes.dropna(inplace=True)
    clothes.reset_index(drop=True, inplace=True)
    clothes.drop(['class_name', 'outfit_id', 'image_path', 'caption'], axis = 1, inplace=True)
    clothes = pd.get_dummies(clothes, dtype=int) # Si on utilise une méthode embeddé, on pourrait préférer ne pas faire de one-hot encoding

    # Generate random user
    users = [rd.sample(range(len(clothes)), 50)]

    return run_recommendation_pipeline(clothes, reco_algo, users, extra_ra_args)