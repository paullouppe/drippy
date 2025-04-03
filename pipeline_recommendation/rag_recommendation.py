import os
import ollama
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

EMBEDDING_PATH = "checkpoints/embedding_clothes.npy"
DATASET_PATH = "checkpoints/clothes_second_augmentation.pkl"
SYSTEM_PROMPT = """You are a helpful reading assistant who answers questions 
        based on snippets of text provided in context. Answer only using the context provided, 
        being as concise as possible. If you're unsure, just say that you don't know.
        Context:
    """

def retriever(query, bdd, df, model, top_k=1):
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


def load_RAG():
    # Load dataset
    df = pd.read_pickle(DATASET_PATH)

    # Load or create embeddings
    embedding = load_embeddings(EMBEDDING_PATH)
    return df, embedding

def call_RAG(query, bdd, df, query_embedding_model_name='all-minilm', conversationnal_model_name='qwen2.5'):
    most_similar_chunks = retriever(query, bdd, df, query_embedding_model_name)

    print("Retrieved:", most_similar_chunks)

    # response = ollama.chat(
    #     model=conversationnal_model_name,
    #     messages=[
    #         {
    #             "role": "system",
    #             "content": SYSTEM_PROMPT
    #             + "\n".join([f"Type of clothes: {chunk['class_name']} Caption: {chunk['caption']} Color: {chunk['baseColour']} Category: {chunk['category']} Usage: {chunk['usage']}" for _, chunk in most_similar_chunks]),            
    #         },
    #         {"role": "user", "content": query},
    #     ],
    # )
    # print("\n\n")
    # print(response["message"]["content"])

    # return most_similar_chunks, response["message"]["content"]


question = "Outfit with a grey shirt"
df, bdd = load_RAG()
# ms_chunks, response = call_RAG(question, bdd, df, 'all-minilm')
call_RAG(question, bdd, df, 'all-minilm')
