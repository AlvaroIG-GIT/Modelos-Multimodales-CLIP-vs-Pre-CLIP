import json
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# 1. Cargar dataset.json
with open('Dataset.json', 'r', encoding='utf-8') as f:
    dataset = json.load(f)

image_paths = [item['image'] for item in dataset]

# 2. Cargar modelo y procesador CLIP
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)
model.eval()

# 3. Definir consulta de texto
query = "A cat standing straight on a rock"
print(f"Consulta de texto: '{query}'")

# 4. Procesar imágenes y texto
images = [Image.open(path).convert('RGB') for path in image_paths]
inputs = processor(text=[query], images=images, return_tensors="pt", padding=True)

# 5. Calcular similitud y encontrar la mejor coincidencia
with torch.no_grad():
    image_features = model.get_image_features(pixel_values=inputs['pixel_values'])
    text_features = model.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    similarity = torch.matmul(image_features, text_features.T).squeeze()
    best_match_idx = similarity.argmax().item()

print(f"Mejor imagen encontrada: {image_paths[best_match_idx]} (Similitud: {similarity[best_match_idx].item():.4f})")