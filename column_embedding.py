import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch



def embed_columns():
    print("Reading listings...")
    listings = pd.read_csv("listings.csv")

    print("Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer('intfloat/multilingual-e5-small', device=device)
    print("Model loaded.")

    columns_to_encode = ['name', 'description']
    embeddings_store = {}

    for col in columns_to_encode:
        text_list = listings[col].fillna("").astype(str).tolist()
        formatted_text_list = [f"query: {text}" for text in text_list]

        print(f"Encoding column: {col}...")
        embeddings = model.encode(formatted_text_list, show_progress_bar=True)
        embeddings_store[col] = embeddings

    np.savez("embeddings.npz", **embeddings_store)


if __name__ == "__main__":
    embed_columns()