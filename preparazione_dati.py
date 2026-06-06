import pandas as pd
import numpy as np
import os
import cv2
from PIL import Image
import pickle

EXCEL_PATH = 'dataset/dataset_laboratorio.xlsx.xlsx'
IMMAGINI_PATH = 'dataset/immagini/'
OUTPUT_PATH = 'modello/'

os.makedirs(OUTPUT_PATH, exist_ok=True)

print("Caricamento dataset Excel...")
df = pd.read_excel(EXCEL_PATH)
print("Colonne trovate:", df.columns.tolist())
print("Numero etichette:", len(df))

COLONNE_PUNTEGGI = [
    'Eleganza dell\'etichetta',
    'Completezza delle informazioni',
    'Coerenza cromatica etichetta/vino',
    'Qualita del design',
    'Attrattivita per i giovani'
]

COLONNE_DESCRIZIONI = [
    'Descrizione eleganza',
    'Descrizione completezza',
    'Descrizione coerenza',
    'Descrizione design',
    'Descrizione attrattivita'
]

df.columns = [c.strip() for c in df.columns]

col_punteggi_reali = []
for col in df.columns:
    col_pulita = col.replace("'", "").replace("à", "a").replace("è", "e").replace("ò", "o").replace("ù", "u").replace("ì", "i")
    for cp in COLONNE_PUNTEGGI:
        cp_pulita = cp.replace("'", "").replace("à", "a").replace("è", "e").replace("ò", "o").replace("ù", "u").replace("ì", "i")
        if col_pulita.lower() == cp_pulita.lower():
            col_punteggi_reali.append(col)
            break

if len(col_punteggi_reali) == 0:
    col_punteggi_reali = [c for c in df.columns if any(k in c.lower() for k in ['eleganza', 'completezza', 'coerenza', 'design', 'attrattiv'])]

print("Colonne punteggi trovate:", col_punteggi_reali)

col_immagine = None
for col in df.columns:
    if 'immagine' in col.lower() or 'img' in col.lower() or 'foto' in col.lower():
        col_immagine = col
        break

if col_immagine is None:
    col_immagine = df.columns[-1]

print("Colonna immagine:", col_immagine)

col_punteggio_medio = None
for col in df.columns:
    if 'medio' in col.lower() or 'complessivo' in col.lower() or 'finale' in col.lower():
        col_punteggio_medio = col
        break

print("Colonna punteggio medio:", col_punteggio_medio)

def estrai_features_immagine(img_path):
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None
        
        img_resized = cv2.resize(img, (128, 128))
        
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        colori_medi = img_rgb.mean(axis=(0, 1))
        colori_std = img_rgb.std(axis=(0, 1))
        
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        contrasto = gray.std()
        
        luminosita = gray.mean()
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        nitidezza = laplacian.var()
        
        hist_r = cv2.calcHist([img_rgb], [0], None, [8], [0, 256]).flatten()
        hist_g = cv2.calcHist([img_rgb], [1], None, [8], [0, 256]).flatten()
        hist_b = cv2.calcHist([img_rgb], [2], None, [8], [0, 256]).flatten()
        
        hist_r = hist_r / hist_r.sum()
        hist_g = hist_g / hist_g.sum()
        hist_b = hist_b / hist_b.sum()
        
        features = np.concatenate([
            colori_medi,
            colori_std,
            [contrasto, luminosita, nitidezza],
            hist_r, hist_g, hist_b
        ])
        
        return features
    except Exception as e:
        print(f"Errore immagine {img_path}: {e}")
        return None

print("\nEstraendo features dalle immagini...")
features_list = []
punteggi_list = []
punteggio_medio_list = []
nomi_list = []
righe_valide = []

for idx, row in df.iterrows():
    nome_img = str(row[col_immagine]).strip()
    img_path = os.path.join(IMMAGINI_PATH, nome_img)
    
    if not os.path.exists(img_path):
        possibili = [f for f in os.listdir(IMMAGINI_PATH) if nome_img.lower() in f.lower()]
        if possibili:
            img_path = os.path.join(IMMAGINI_PATH, possibili[0])
        else:
            print(f"Immagine non trovata: {nome_img}")
            continue
    
    features = estrai_features_immagine(img_path)
    if features is None:
        continue
    
    punteggi_riga = []
    for col in col_punteggi_reali:
        try:
            val = float(str(row[col]).replace(',', '.'))
            punteggi_riga.append(val)
        except:
            punteggi_riga.append(0.0)
    
    if len(punteggi_riga) == 0:
        continue
    
    if col_punteggio_medio:
        try:
            pm = float(str(row[col_punteggio_medio]).replace(',', '.'))
        except:
            pm = np.mean(punteggi_riga)
    else:
        pm = np.mean(punteggi_riga)
    
    nome_vino = str(row.get('Nome del vino', str(row.iloc[1]))).strip()
    
    features_list.append(features)
    punteggi_list.append(punteggi_riga)
    punteggio_medio_list.append(pm)
    nomi_list.append(nome_vino)
    righe_valide.append(idx)

print(f"\nImmagini processate con successo: {len(features_list)}")

X = np.array(features_list)
y = np.array(punteggio_medio_list)
y_categorie = np.array(punteggi_list)

np.save(os.path.join(OUTPUT_PATH, 'X.npy'), X)
np.save(os.path.join(OUTPUT_PATH, 'y.npy'), y)
np.save(os.path.join(OUTPUT_PATH, 'y_categorie.npy'), y_categorie)

with open(os.path.join(OUTPUT_PATH, 'metadati.pkl'), 'wb') as f:
    pickle.dump({
        'nomi': nomi_list,
        'colonne_punteggi': col_punteggi_reali,
        'col_punteggio_medio': col_punteggio_medio,
        'col_immagine': col_immagine
    }, f)

print("\nDati salvati nella cartella 'modello/'")
print("Punteggio medio nel dataset:", round(y.mean(), 2))
print("Range punteggi:", round(y.min(), 2), "-", round(y.max(), 2))
print("\nPreparazione dati completata!")