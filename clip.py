import json
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np

# 1. Cargar dataset.json
with open('Dataset.json', 'r', encoding='utf-8') as f:
    dataset = json.load(f)

image_paths = [item['image'] for item in dataset]
captions = [item['caption'] for item in dataset]

print(f"Cargadas {len(image_paths)} imágenes y {len(captions)} textos.")

# 2. Cargar modelo y procesador CLIP unificado
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)
model.eval()

# 3. Cargar imágenes y procesarlas junto a los textos
images = [Image.open(path).convert('RGB') for path in image_paths]
inputs = processor(text=captions, images=images, return_tensors="pt", padding=True)

# 4. Obtener embeddings en el espacio vectorial compartido
with torch.no_grad():
    image_features = model.get_image_features(pixel_values=inputs['pixel_values'])
    text_features = model.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])

    # Normalizar vectores
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Matriz de similitud cruzada
    similarity_matrix = torch.matmul(image_features, text_features.T).numpy()

print("\n--- Resultados CLIP ---")
print("Dimensiones matriz de similitud:", similarity_matrix.shape)
print("Diagonal principal (correspondencia exacta imagen-texto):")
print(np.diagonal(similarity_matrix))