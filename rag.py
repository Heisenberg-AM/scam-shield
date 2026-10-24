import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import streamlit as st

# Constants
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
MIN_SIMILARITY = 0.25
KB_SCAM_MESSAGES_PATH = "data/kb_scam_messages.csv"
KB_TRUSTED_INFO_PATH = "data/kb_trusted_info.csv"

def build_index():
    """
    Reads knowledge base CSVs, embeds texts, and builds a FAISS index.
    Returns the loaded dataframe, the sentence transformer model, and the FAISS index.
    """
    # Load knowledge base data
    df_scam = pd.read_csv(KB_SCAM_MESSAGES_PATH)
    df_trusted = pd.read_csv(KB_TRUSTED_INFO_PATH)

    # Combine datasets
    df_combined = pd.concat([df_scam, df_trusted], ignore_index=True)

    # Load embedding model
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Embed texts with L2 normalization for cosine similarity
    texts = df_combined['text'].tolist()
    embeddings = model.encode(texts, normalize_embeddings=True)

    # Build FAISS IndexFlatIP (Inner Product = Cosine Similarity for normalized vectors)
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(np.array(embeddings, dtype=np.float32))

    return df_combined, model, index

@st.cache_resource
def load_resources():
    """
    Wrapper for build_index to cache resources in Streamlit.
    Builds the FAISS index in memory at startup.
    """
    return build_index()

def retrieve_evidence(message, top_k=5):
    """
    Embeds the input message and searches the FAISS index for similar evidence.
    Returns a list of dictionaries with text, source_type, and similarity score.
    Drops results below the MIN_SIMILARITY threshold.
    """
    # Retrieve cached resources
    df_combined, model, index = load_resources()

    # Embed the query message
    query_embedding = model.encode([message], normalize_embeddings=True)

    # Search FAISS index
    similarities, indices = index.search(np.array(query_embedding, dtype=np.float32), top_k)

    evidence = []
    for sim, idx in zip(similarities[0], indices[0]):
        if sim >= MIN_SIMILARITY:
            row = df_combined.iloc[idx]
            evidence.append({
                "text": row['text'],
                "source_type": row['source_type'],
                "similarity": float(sim)
            })

    return evidence
