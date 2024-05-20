import requests
import io
from PIL import Image

API_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
headers = {"Authorization": "Bearer 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"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.content

# Liste oluşturarak 7 farklı resim için payload'ları oluştur
payloads = [{"inputs": f"green dress {i}"} for i in range(1, 8)]

# Resimleri üretmek için döngü
for i, payload in enumerate(payloads):
    image_bytes = query(payload)
    image = Image.open(io.BytesIO(image_bytes))
    image_name = f"image_{i+1}.jpg"
    image.save(f"generated_images/{image_name}")  # Üretilen resimleri kaydet

print("Resimler başarıyla üretildi.")
