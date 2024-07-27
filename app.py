import streamlit as st
from PIL import Image
from authtoken import auth_token

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

# Initialize the Stable Diffusion pipeline
modelid = "runwayml/stable-diffusion-v1-5"
device = "cuda"
pipe = StableDiffusionPipeline.from_pretrained(modelid, torch_dtype=torch.float16, use_auth_token=auth_token) 
pipe.to(device)

# Streamlit app
st.title("IMAGE GENERATOR")
st. write("This app uses the Stable Diffusion model to generate high-quality images based on your input, made By Fakhrus")

prompt = st.text_input("Enter your prompt", "")
guidance_scale = st.slider("Guidance Scale", min_value=1.0, max_value=20.0, value=8.5)

if st.button("Generate"):
    if prompt:
        with st.spinner('Generating image...'):
            with autocast(device):
                image = pipe(prompt, guidance_scale=guidance_scale).images[0]
            image.save('generatedimage.png')
            st.image('generatedimage.png', caption='Generated Image', use_column_width=True)
    else:
        st.warning("Please enter a prompt to generate an image.")
