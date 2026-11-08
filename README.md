# Práctica 4: Modelos Multimodales (CLIP vs Pre-CLIP)

Repositorio para la Práctica de la asignatura **Sistemas Interactivos Inteligentes**. El proyecto evalúa y compara el rendimiento y la alineación de espacios vectoriales entre modelos independientes (*Pre-CLIP*) y modelos multimodales conjuntos con aprendizaje contrastivo (*CLIP*), incluyendo recuperación de imágenes (*Zero-Shot*).

---

## Estructura del Repositorio

* `Dataset/`: Carpeta con las 20 imágenes organizadas en 4 categorías (*Aves*, *Felinos*, *Peces* y *Reptiles*).
* `Dataset.json`: Archivo único de metadatos y *captions* descriptivos en inglés.
* `pre_clip.py`: Script para la extracción de características independientes (ResNet18 para imágenes y Sentence-Transformers para textos).
* `clip.py`: Script para la generación de la matriz de similitud cruzada $20 \times 20$ utilizando la arquitectura unificada CLIP.
* `search.py`: Script de búsqueda y recuperación de imágenes por lenguaje natural (*Zero-Shot* / Retrieval).
* `Dockerfile` & `requirements.txt`: Configuración para la contenerización y aislamiento del entorno de ejecución.

---

## Ejecución con Docker

Para ejecutar los scripts garantizando la reproducibilidad en cualquier sistema, se utiliza Docker:

1. **Construir la imagen del contenedor:**
   ```bash
   docker build -t practica4-env .