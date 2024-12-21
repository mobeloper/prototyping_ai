import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

st.title("Stable Diffusion: An Exploration")
st.markdown("Explore the capabilities of the CompVis latent text-to-image Stable Diffusion 1.4 model.")

st.sidebar.header("Input Parameters")
prompt = st.sidebar.text_input("Text Prompt", "")
negative_prompt = st.sidebar.text_input("Negative Prompt", "")
num_inference_steps = st.sidebar.slider("Number of Inference Steps", 10, 50, 25)
guidance_scale = st.sidebar.slider("Temperature", 5.0, 20.0, 7.5)
height = st.sidebar.number_input("Image Height (pixels)", min_value=128, max_value=1024, value=512, step=128)
width = st.sidebar.number_input("Image Width (pixels)", min_value=128, max_value=1024, value=512, step=128)
generate_button = st.sidebar.button("Generate Image")

@st.cache_resource
def load_pipeline():
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipe

pipeline = load_pipeline()

if generate_button:
    with st.spinner("Generating image..."):
        image = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width
        ).images[0]
        st.image(image, caption="Generated Image", use_column_width=True)

