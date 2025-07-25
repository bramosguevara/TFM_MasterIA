# TFM Máster en Inteligencia Artificial VIU 2024-2025

Este repositorio contiene el código, los recursos y los datos necesarios para reproducir el Trabajo de Fin de Máster titulado **"Aplicación de IA Generativa en el diseño de videojuegos"**.

El objetivo principal del proyecto es aplicar técnicas de IA Generativa para la **creación automática de cartas visuales y narrativas**, simulando mecánicas inspiradas en el juego de Clash Royale.

---

## Estructura del Proyecto

### 1. **Fases del desarrollo del modelo de IA Generativa**

El desarrollo se divide en tres fases principales, reflejadas en los notebooks:

- `IAGenFirstPhase.ipynb`:  
  🔹 Limpieza y preparación de los datos base (datasets de cartas y descripciones narrativas).  
  🔹 Conversión, normalización y eliminación de inconsistencias.

- `IAGenSecondPhase.ipynb`:  
  🔹 Generación estructural y narrativa usando modelos de lenguaje (NLP).  
  🔹 Creación de descripciones de cartas, habilidades y estructuras de juego.

- `IAGenThirdPhase.ipynb`:  
  🔹 Generación visual utilizando modelos de difusión (stable diffusion o similares).  
  🔹 Composición final de las cartas con texto e imagen.

---

### 2. **Aplicación interactiva**

- `StreamlitApp.py`:  
  Aplicación desarrollada en **Streamlit** para visualizar, probar y generar nuevas cartas de manera interactiva a partir del modelo entrenado.

---

### 3. **Datasets utilizados**

- `clash_dataset_cleaned.csv`:  
  Dataset principal con datos limpios usados para entrenamiento (estructuras narrativas y atributos de cartas).

- `clash_test.csv`:  
  Subconjunto usado para pruebas durante el entrenamiento.

- `clash_train.csv`:  
  Subconjunto usado para entrenar los modelos.

- `clash_wiki_dataset.csv`:  
  Dataset extraído de una wiki temática como base de conocimiento semiestructurado.

---

### 4. **Resultados generados**

- `diffusion_gallery.png`:  
  Galería de ejemplos visuales generados por el modelo de difusión.

- `final_card_01.png` a `final_card_06.png`:  
  Ejemplos de **cartas completas generadas automáticamente**, incluyendo imagen, nombre, tipo, y descripción narrativa.

---
