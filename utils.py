import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

@st.cache_data
def load_data():
    """Loads user profiles and prepares combined text fields."""
    df = pd.read_csv("users.csv")
    df["combined_text"] = (
        df["professional_summary"].fillna("") + " " + 
        df["about_me"].fillna("") + " " + 
        df["interests"].fillna("")
    ).str.lower()
    return df

@st.cache_resource
def get_similarity_matrix(df):
    """Computes TF-IDF and Cosine Similarity matrix."""
    tfidf = TfidfVectorizer(stop_words="english")
    text_vectors = tfidf.fit_transform(df["combined_text"])
    return cosine_similarity(text_vectors)

def get_mbti_score(mbti1, mbti2):
    """Calculates personality compatibility based on MBTI rules."""
    if mbti1 == mbti2:
        return 1.0
    elif mbti1[0] != mbti2[0] and mbti1[1] == mbti2[1]:
        return 0.8
    else:
        return 0.5

def calculate_top_matches(user_id, df, similarity_matrix, top_n=5):
    """Calculates weighted compatibility scores for a given user."""
    idx = df.index[df["user_id"] == user_id][0]
    user_row = df.iloc[idx]
    
    scores = []
    for j, row in df.iterrows():
        if j != idx:
            text_sim = similarity_matrix[idx, j]
            mbti_score = get_mbti_score(user_row["mbti"], row["mbti"])
            loc_score = 1.0 if user_row["location"] == row["location"] else 0.3
            
            # Weighted Scoring Formula: 60% Text, 30% MBTI, 10% Location
            total_score = (0.6 * text_sim) + (0.3 * mbti_score) + (0.1 * loc_score)
            scores.append({
                "user_id": row["user_id"],
                "name": row["name"],
                "profession": row["profession"],
                "location": row["location"],
                "mbti": row["mbti"],
                "score": round(total_score * 100, 2)
            })
    
    # Sort by highest score
    scores.sort(key=lambda x: x["score"], reverse=True)
    return user_row, scores[:top_n]