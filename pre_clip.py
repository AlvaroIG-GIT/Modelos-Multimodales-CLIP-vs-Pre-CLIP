import json
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 1. Cargar dataset.json
with open('Dataset.json', 'r', encoding='utf-8') as f:
    dataset = json.load(f)

image_paths = [item['image'] for item in dataset]
captions = [item['caption'] for item in dataset]

print(f"Cargadas {len(image_paths)} imágenes y {len(captions)} textos.")

# 2. Extraccion de embeddings de imagen con ResNet18
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
# Quitamos la capa de clasificacion final para quedarnos con el vector de caracteristicas
feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])
feature_extractor.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image_embeddings = []
with torch.no_grad():
    for path in image_paths:
        img = Image.open(path).convert('RGB')
        tensor = transform(img).unsqueeze(0)
        feat = feature_extractor(tensor).squeeze() # Vector de 512 dimensiones
        image_embeddings.append(feat.numpy())

image_embeddings = np.array(image_embeddings)

# 3. Extraccion de embeddings de texto con Sentence-Transformers
text_model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased')
text_embeddings = text_model.encode(captions)

# 4. Calcular similitudes (Cosine Similarity)
img_sim = cosine_similarity(image_embeddings)
txt_sim = cosine_similarity(text_embeddings)

print("\n--- Resultados Pre-CLIP ---")
print("Dimensiones embedding imagen:", image_embeddings.shape)
print("Dimensiones embedding texto:", text_embeddings.shape)
print("Similitud cruzada calculada. Como usan espacios vectoriales separados, no coinciden directamente.")