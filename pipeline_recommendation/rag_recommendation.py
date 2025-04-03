import os
import ollama
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def SYSTEM_PROMPT(user_query, outfit):
    return f"""
    You are a fashion assistant helping users choose outfits based on their specific needs.
    The user has asked:
    "{user_query}"

    The following outfit has been recommended:
    {outfit}

    Your task is to explain why this outfit fits the user’s request, referring to the occasion, style, weather, season, practicality, fashion trends, and personal preferences if available. Make the explanation feel personal and thoughtful.
    
    Keep it concise!
    
    Use clear, friendly language that balances expertise and approachability. Justify the recommendation with relevant fashion reasoning (e.g., color palette, fabric, cut, accessories, context-appropriateness).
    """

def retriever(query, bdd, df, model, top_k=20):
    try:
        processed_query = ' '.join(map(str, [query]))
        query_embedding = ollama.embeddings(model=model, prompt=processed_query)["embedding"]

        similarities = cosine_similarity(
            [query_embedding],  
            bdd 
        ).flatten()

        top_indices = similarities.argsort()[::-1][:top_k]

        results = [(similarities[i], df.iloc[i].to_dict()) for i in top_indices]
        
        return results

    except Exception as e:
        print(f"Error during search: {e}")
        return []

def load_embeddings(embedding_path):
    if os.path.exists(embedding_path):
        print(f"Loading embeddings from {embedding_path}...")
        bdd = np.load(embedding_path, allow_pickle=True)
    else:
        print("Run data integration to generate embeddings.")
    return bdd


def outfit_adaptation():
    pass


def load_RAG(dataset_path, embedding_path, user_clothes_dataset_path, user_clothes_embedding_path):
    # Load dataset
    df = pd.read_pickle(dataset_path)
    user_clothes_df = pd.read_csv(user_clothes_dataset_path)

    # Load or create embeddings
    embedding = load_embeddings(embedding_path)
    user_clothes_embedding = load_embeddings(user_clothes_embedding_path)
    return df, embedding, user_clothes_df, user_clothes_embedding

from pprint import pprint
def call_RAG(query, outfit_clothes_embedding, user_clothes_embedding, clothes_df, user_clothes_df, query_embedding_model_name='all-minilm', conversationnal_model_name='qwen2.5'):
    most_similar_chunks = retriever(query, outfit_clothes_embedding, clothes_df, query_embedding_model_name)

    # print("Retrieved:", most_similar_chunks)

    pondered_outfits = {}

    for (score, cloth) in most_similar_chunks:
        if pondered_outfits.get(cloth["outfit_id"]):
            pondered_outfits[cloth["outfit_id"]] += score
        else:
            pondered_outfits[cloth["outfit_id"]] = score
        
    selected_outfit = max(pondered_outfits, key=pondered_outfits.get)

    # Selected outfit is the on for the outfit recommendation
    selected_clothes = pd.DataFrame([cl for _, cl in clothes_df.iterrows() if cl.outfit_id == selected_outfit])
    
    selected_clothes.drop(['outfit_id', 'image_path'], axis = 1, inplace=True)

    most_similar_user_clothes = []
    for _, cl in selected_clothes.iterrows():
        sim = retriever(cl, user_clothes_embedding, user_clothes_df, query_embedding_model_name, top_k=1)
        most_similar_user_clothes.append(sim)

    response = ollama.chat(
        model=conversationnal_model_name,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT(query, "\n".join([
                    f"Type: {chunk['articleType']} Color: {chunk['baseColour']} Gender: {chunk['gender']} Category: {chunk['masterCategory']} Display name: {chunk['productDisplayName']} Subcategory: {chunk['subCategory']} Usage: {chunk['usage']}"
                    for sublist in most_similar_user_clothes
                    for score, chunk in sublist
                ]))
            },
            {"role": "user", "content": query},
        ],
    )
    print("\n\n")
    print(response["message"]["content"])

    return most_similar_user_clothes, response["message"]["content"]
