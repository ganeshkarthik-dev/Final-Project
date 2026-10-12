import streamlit as st
from utils import load_data, get_similarity_matrix, calculate_top_matches

# Page Setup
st.set_page_config(page_title="Intelligent Hybrid Matchmaker", page_icon="⚡", layout="wide")

# Function to load external CSS file for cards and layout
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("styles.css")

# Load Data & Models
df_users = load_data()
similarity_matrix = get_similarity_matrix(df_users)

# Sidebar UI
st.sidebar.markdown("### 🎛️ Navigation")
selected_user_id = st.sidebar.selectbox("Select Active User ID:", df_users["user_id"].unique())

# Fetch Data for Selected User
user_row, top_matches = calculate_top_matches(selected_user_id, df_users, similarity_matrix)

# Main Page Layout using native Streamlit text rendering (guaranteed visibility)
st.title("⚡ Intelligent Hybrid Recommendation System")
st.markdown("##### Semantic NLP text analysis, MBTI profiling, and adaptive matching.")
st.write("")

# Active User Profile Card
st.markdown(f"""
    <div class="profile-card">
        <h3>👤 Active Profile: {user_row['name']}</h3>
        <p><b>Role:</b> {user_row['profession']} | <b>Location:</b> {user_row['location']} | <b>MBTI:</b> {user_row['mbti']}</p>
        <hr style="border-color: #334155;">
        <p><b>Summary:</b> {user_row['professional_summary']}</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("### 🌟 Top Recommended Matches")

# Render Recommendations
for i, match in enumerate(top_matches, start=1):
    st.markdown(f"""
        <div class="match-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h4 style="margin: 0; color: #F8FAFC;">#{i} — {match['name']}</h4>
                <span class="metric-badge">Match: {match['score']}%</span>
            </div>
            <p style="margin: 6px 0 0 0; color: #94A3B8;">
                <b>Role:</b> {match['profession']} | <b>Location:</b> {match['location']} | <b>MBTI:</b> {match['mbti']}
            </p>
        </div>
    """, unsafe_allow_html=True)