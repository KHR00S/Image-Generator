from flask import Flask, request, jsonify, send_file
from PIL import Image
from io import BytesIO
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from authtoken import auth_token

app = Flask(__name__)

modelid = "stabilityai/stable-diffusion-3-mediums"
device = "cuda"
pipe = StableDiffusionPipeline.from_pretrained(modelid, torch_dtype=torch.float16, use_auth_token=auth_token)
pipe.to(device)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get("prompt")
    guidance_scale = data.get("guidance_scale", 8.5)

    if not prompt:
        return jsonify({"error": "Please enter a prompt"}), 400

    with autocast(device):
        image = pipe(prompt, guidance_scale=guidance_scale).images[0]

    img_io = BytesIO()
    image.save(img_io, 'PNG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
