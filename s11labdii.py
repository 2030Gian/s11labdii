# conda install conda-forge::face_recognition
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rtree import index
import face_recognition
import pickle

DATASET_PATH = "/home/alissgian2030/PycharmProjects/s11labdii/lfw_funneled"

coleccion = []
for path in glob.iglob(os.path.join(DATASET_PATH, "**", "*.jpg")):

    person = path.split(os.path.sep)[-2]

    coleccion.append({"person":person, "path": path})

coleccion = pd.DataFrame(coleccion)
coleccion.head(10)
#print(len(coleccion))

def mostrarFotos(coleccion, indices):
    plt.figure(figsize=(16,10))
    i = 0
    for idx in indices:
        img = plt.imread(coleccion.path.iloc[idx])
        plt.subplot(4, 4, i+1)
        plt.imshow(img)
        plt.title(coleccion.person.iloc[idx]+str(img.shape))
        plt.xticks([])
        plt.yticks([])
        i += 1
    plt.tight_layout()
    plt.show()

#indices = np.random.randint(0, len(coleccion), 5)
indices = list(range(0, 16))
mostrarFotos(coleccion, indices)

def get_face_embeddings(image_path):
    try:
        image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(image)

        if face_encodings:
            return face_encodings[0]
        else:
            return None
    except Exception as e:
        print(f"Error procesando {image_path}: {e}")
        return None

print("Iniciando vectorización...")
coleccion['embedding'] = coleccion['path'].apply(get_face_embeddings)

coleccion_clean = coleccion.dropna(subset=['embedding']).copy()

coleccion_clean['embedding'] = coleccion_clean['embedding'].apply(lambda x: x.tolist())

print(f"Vectorización completa. Se procesaron {len(coleccion_clean)} rostros.")

# Guardamos un pkl DE RESPALDO, aunque lo principal irá a PostgreSQL
with open('face_embeddings_clean.pkl', 'wb') as f:
    pickle.dump(coleccion_clean, f)

coleccion_clean.head()