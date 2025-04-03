# Import Libraries

import streamlit as st
import os
import numpy as np
import pandas as pd
import requests
import pickle

api_key = os.getenv("apikey")

thirty_thou_dict = pickle.load(open("thirty_thou_redu.pkl", "rb"))
thirty_thou = pd.DataFrame(thirty_thou_dict)
thirtythou_new_dict = pickle.load(open("thirtythou_new_redu.pkl", "rb"))
thirtythou_new = pd.DataFrame(thirtythou_new_dict)

# Replace sklearn's cosine_similarity with NumPy
def cosine_similarity_np(matrix):
    numerator = matrix @ matrix.T
    norms = np.linalg.norm(matrix, axis=1)
    denominator = np.outer(norms, norms)
    return numerator / denominator

similarity = cosine_similarity_np(thirtythou_new.values)  # Use .values to get NumPy array

def recommend(song, artist):
    index = thirty_thou[(thirty_thou['track_name'] == song) & (thirty_thou['track_artist'] == artist)].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    songs = []
    artists = []
    for i in distances[1:6]:
        songs.append(thirty_thou.iloc[i[0]]['track_name'])
        artists.append(thirty_thou.iloc[i[0]]['track_artist'])
    return songs, artists

# Frontend
st.write(""" <p> Song recommendation engine suggests songs based on your interest </p>""", unsafe_allow_html=True)
st.write("##")
my_expander = st.expander("Tap to Select a Song 🌐️")
selected_song_name = my_expander.selectbox("", thirty_thou["track_name"].values)
selected_artist_name = my_expander.selectbox("", thirty_thou["track_artist"].values)

if my_expander.button("Recommend"):
    st.text("Here are few Recommendations..")
    st.write("#")
    songs, artists = recommend(selected_song_name, selected_artist_name)
    col1, col2, col3, col4, col5 = st.columns(5)
    cols = [col1, col2, col3, col4, col5]
    for i in range(0, 5):
        with cols[i]:
            st.write(f' <b style="color:#E50914"> {songs[i]} </b>', unsafe_allow_html=True)
            st.write(f' <b style="color:#E50914"> {artists[i]} </b>', unsafe_allow_html=True)
