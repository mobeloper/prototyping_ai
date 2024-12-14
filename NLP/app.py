import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Title of the app
st.title("Word Cloud Generator")

# Input text
st.sidebar.header("Word Cloud Settings")
input_text = st.sidebar.text_area(
    "Enter text to generate the word cloud:",
    "Streamlit is an awesome tool for building data apps quickly!"
)

background_color = st.sidebar.color_picker("Pick a background color:", "#ffffff")
max_words = st.sidebar.slider("Maximum number of words:", min_value=10, max_value=200, value=100, step=10)

if st.sidebar.button("Generate Word Cloud"):
    wordcloud = WordCloud(
        background_color=background_color,
        max_words=max_words,
        width=800,
        height=400
    ).generate(input_text)

    # Display the Word Cloud
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)
else:
    st.write("Use the sidebar to configure and generate your word cloud!")
