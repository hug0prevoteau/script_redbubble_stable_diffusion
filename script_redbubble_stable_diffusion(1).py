import torch
from torch import autocast
from tqdm.auto import tqdm
from diffusers import StableDiffusionImg2ImgPipeline
import requests
from io import BytesIO
from PIL import Image
import re
import json
from huggingface_hub import HfApi, HfFolder

TOKEN = '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'
DEVICE = "cuda"
MODEL_PATH ="CompVis/stable-diffusion-v1-4"
JSON_PATH = "/Users/hugofp/Downloads/results.json"

def get_image(url):
    response = requests.get(url)
    init_img = Image.open(BytesIO(response.content)).convert("RGB")
    init_img = init_img.resize((512, 512))
    return init_img

def clean_prompt(prompt):
    cleaned_prompt = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', prompt, flags=re.MULTILINE)
    cleaned_prompt = re.sub("[^a-zA-Z0-9. ]", "",cleaned_prompt).replace("  ", " ").replace(' - ', " ")
    return cleaned_prompt

def api_lexica(img_url):
    r = requests.get('https://lexica.art/api/v1/search?q='+img_url)
    return r.json()

def generate_images(data):
    for i in range(len(data)):
        init_img = get_image(data[i]["image_url"])
        prompt = data[i]["title"] + ". " + data[i]["description"] + ". " + " ".join(data[i]["tags"])
        prompt = clean_prompt(prompt)
        try:
            prompt += api_lexica(data[i]["image_url"])['images'][0]['prompt']
        except:
            pass
        with autocast("cuda"): 
            image = pipe(prompt=prompt, init_image=init_img, strength=0.60, guidance_scale=7.5, generator=generator, num_inference_steps=70).images[0]
        image.save(f"generated_images/{i}.jpg")

api=HfApi()
api.set_access_token(TOKEN)
folder = HfFolder()
folder.save_token(TOKEN)

f = open(JSON_PATH, encoding="utf8")
data = json.load(f)

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    MODEL_PATH,
    revision="fp16", 
    torch_dtype=torch.float16,
    use_auth_token=True
)
pipe = pipe.to(DEVICE)

generator = torch.Generator(device=DEVICE).manual_seed(1024)
generate_images(data)