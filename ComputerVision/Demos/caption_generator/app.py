import os
import base64
import streamlit as st
from groq import Groq
from openai import OpenAI
from PIL import Image

# Setting Page Title, Page Icon, and Layout Size
st.set_page_config(
    page_title="Pandita - Multilingual Image Caption Generator",
    page_icon="images/pandita-favicon.webp",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Function to load external CSS
def load_css(file_name):
    with open(file_name, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load the CSS file
load_css("style.css")

@st.cache_data(show_spinner=False)
def get_base64_of_bin_file(png_file):
    with open(png_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def build_markup_for_logo(
    png_file, background_position="45px 25px", margin_top="0%", image_width="100px", image_height=""
):
    binary_string = get_base64_of_bin_file(png_file)
    return """
            <style>
                [data-testid=stHeader] {
                    background-image: url("data:image/png;base64,%s");
                    background-repeat: no-repeat;
                    background-position: %s;
                    margin-top: %s;
                    background-size: %s %s;
                    border-bottom:2px solid #F5CEAE;
                    height:75px;
                    background-color: #FCFBF0;
                }
            </style>
            """ % (
        binary_string,
        background_position,
        margin_top,
        image_width,
        image_height,
    )

def add_logo(png_file):
    logo_markup = build_markup_for_logo(png_file)
    st.markdown(logo_markup, unsafe_allow_html=True)

add_logo("images/Logomark.png")

# Initialize the API keys for Groq and OpenAI services
api_key_g = os.getenv('GROQ_API_KEY')
client = Groq(api_key=api_key_g)

api_key_a = os.getenv('OPENAI_API_KEY')
openai_client = OpenAI(api_key=api_key_a)

# List of languages for translation with language codes
languages = {
    "Spanish": "es",
    "German": "de",
    "Portuguese": "pt",
    "French": "fr",
    "Italian": "it",
    "Russian": "ru",
    "Japanese": "ja",
    "Chinese": "zh",
    "Korean": "ko",
    "Hindi": "hi",
}

# Function to encode the image as base64 string
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to generate an English caption from the provided image using Groq's Llama 3.2-Vision
def generate_caption(image_filepath: str) -> str:
    base64_image = encode_image(image_filepath)
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "You are a knowledgeable image caption generation assistant. Write a brief, accurate caption for what you see in the image in a few sentences."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
        model="llama-3.2-11b-vision-preview",
        temperature=0.2,
        max_tokens=256,
        stream=False,
    )
    return chat_completion.choices[0].message.content

# Function to translate text from one language to another using Llama 3.1 model
def groq_translate(query: str, from_language: str, to_language: str) -> str:
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": f"You are a knowledgeable translation assistant that translates text from {from_language} to {to_language}. "
                           f"You will only reply with the translated text in {to_language}, and nothing else."
            },
            {
                "role": "user",
                "content": f"Translate the following from {from_language} to {to_language}: '{query}'"
            }
        ],
        model="llama-3.2-1b-preview",
        temperature=0.2,
        max_tokens=256,
        stream=False,
    )
    return chat_completion.choices[0].message.content

# Function to convert the translated text to speech using OpenAI TTS1
def generate_speech(text: str) -> str:
    response = openai_client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text,
    )
    output_filename = "output.mp3"
    response.stream_to_file(output_filename)
    return output_filename

# Streamlit Application
def main():
    # Initialize session state
    if 'image_path' not in st.session_state:
        st.session_state.image_path = None
    if "selected_language" not in st.session_state:
        st.session_state.selected_language = None
    if 'caption_text' not in st.session_state:
        st.session_state.caption_text = None
    if 'translated_text' not in st.session_state:
        st.session_state.translated_text = None
    if 'tts_audio_file' not in st.session_state:
        st.session_state.tts_audio_file = None

    st.title("Multilingual Image Caption Generator")

    # Select input method outside the form
    # input_method = st.radio("Select input method:", ["Upload Image", "Capture Image"])

    with st.form("caption_form"):
        # Initialize variables
        uploaded_image = None
        # captured_image = None

        # if input_method == "Upload Image":
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        # elif input_method == "Capture Image":
        #     captured_image = st.camera_input("Capture an image using your webcam")

        selected_language = st.selectbox("Select language to translate to:", list(languages.keys()))
        submit_button = st.form_submit_button("Generate Caption")

    # Process the image and display results
    if submit_button:
        try:
            # Check if an image was provided
            # if not uploaded_image and not captured_image:
            if not uploaded_image :
                # Clear session state variables if an error occurs
                st.session_state.image_path = None
                st.session_state.caption_text = None
                st.session_state.translated_text = None
                st.session_state.tts_audio_file = None
                raise ValueError("No image uploaded or captured. Please provide an image.")

            with st.spinner("Processing..."):
                # Save the uploaded or captured image temporarily
                temp_image_path = "temp_image.jpg"
                # if input_method == "Upload Image" and uploaded_image:
                if uploaded_image:
                    with open(temp_image_path, "wb") as f:
                        f.write(uploaded_image.getbuffer())
                # elif input_method == "Capture Image" and captured_image:
                #     with open(temp_image_path, "wb") as f:
                #         f.write(captured_image.read())

                st.session_state.image_path = temp_image_path
                st.session_state.selected_language = selected_language
                st.session_state.caption_text = generate_caption(temp_image_path)
                st.session_state.translated_text = groq_translate(
                    st.session_state.caption_text, "English", st.session_state.selected_language
                )
                st.session_state.tts_audio_file = generate_speech(st.session_state.translated_text)

        except ValueError as ve:
            st.error(f"Error: {ve}")
        except Exception as e:
            # Clear session state variables in case of unexpected errors
            st.session_state.image_path = None
            st.session_state.caption_text = None
            st.session_state.translated_text = None
            st.session_state.tts_audio_file = None
            st.error(f"An unexpected error occurred: {e}")

    # Display results if available in session state
    if st.session_state.image_path:
        st.subheader("Uploaded Image")
        st.image(st.session_state.image_path)
    if st.session_state.caption_text and st.session_state.translated_text:
        st.write("### Captions")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**English**")
            st.text_area("Transcribed Text:", st.session_state.caption_text, height=200)
        with col2:
            st.write(f"**{st.session_state.selected_language}**")
            st.text_area("Translated Text:", st.session_state.translated_text, height=200)

        if st.session_state.tts_audio_file:
            st.write("### Translated Caption Audio Output")
            st.audio(st.session_state.tts_audio_file, format="audio/mp3")
            with open(st.session_state.tts_audio_file, "rb") as audio_file:
                audio_data = audio_file.read()
            st.download_button(
                label="Download Translated Caption Audio",
                data=audio_data,
                file_name="translated_caption.mp3",
                mime="audio/mpeg"
            )

if __name__ == "__main__":
    main()


