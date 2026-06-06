import numpy as np
import pickle
import os
import cv2
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime

MODELLO_PATH = 'modello/'

def carica_modelli():
    with open(os.path.join(MODELLO_PATH, 'modello_globale.pkl'), 'rb') as f:
        modello_globale = pickle.load(f)
    with open(os.path.join(MODELLO_PATH, 'modello_categorie.pkl'), 'rb') as f:
        modello_categorie = pickle.load(f)
    with open(os.path.join(MODELLO_PATH, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODELLO_PATH, 'metadati.pkl'), 'rb') as f:
        metadati = pickle.load(f)
    return modello_globale, modello_categorie, scaler, metadati

def estrai_features_immagine(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Impossibile aprire immagine: {img_path}")

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
        colori_medi, colori_std,
        [contrasto, luminosita, nitidezza],
        hist_r, hist_g, hist_b
    ])
    return features

def genera_descrizione_visiva(punteggi, colonne, punteggio_globale):
    parole = []

    for i, col in enumerate(colonne):
        p = punteggi[i]
        col_lower = col.lower()

        if p >= 9.5:
            if 'eleganza' in col_lower:
                parole.append('estremamente elegante')
            elif 'design' in col_lower:
                parole.append('design eccezionale')
            elif 'coerenza' in col_lower:
                parole.append('cromaticamente perfetta')
            elif 'attrattiv' in col_lower:
                parole.append('molto attrattiva per i giovani')
            elif 'completezza' in col_lower:
                parole.append('informazioni complete')
        elif p >= 9.0:
            if 'eleganza' in col_lower:
                parole.append('elegante')
            elif 'design' in col_lower:
                parole.append('design raffinato')
            elif 'coerenza' in col_lower:
                parole.append('cromaticamente coerente')
            elif 'attrattiv' in col_lower:
                parole.append('attrattiva')
            elif 'completezza' in col_lower:
                parole.append('ben strutturata')
        elif p >= 8.0:
            if 'eleganza' in col_lower:
                parole.append('discreta eleganza')
            elif 'design' in col_lower:
                parole.append('design buono')
            elif 'coerenza' in col_lower:
                parole.append('coerenza cromatica')
        else:
            if 'eleganza' in col_lower:
                parole.append('eleganza migliorabile')
            elif 'design' in col_lower:
                parole.append('design da rivedere')

    if punteggio_globale >= 9.5:
        parole.extend(['premium', 'lussuosa', 'memorabile'])
    elif punteggio_globale >= 9.0:
        parole.extend(['moderna', 'raffinata'])
    elif punteggio_globale >= 8.0:
        parole.extend(['buona qualita', 'classica'])
    else:
        parole.extend(['tradizionale', 'semplice'])

    return ', '.join(parole)

def classifica_etichetta(punteggio):
    if punteggio >= 9.5:
        return 'Altamente persuasiva / virale'
    elif punteggio >= 9.0:
        return 'Molto persuasiva'
    elif punteggio >= 8.0:
        return 'Moderatamente persuasiva'
    else:
        return 'Poco persuasiva'

def consiglio_scaffale(punteggio):
    if punteggio >= 9.5:
        return ('Posizione periferica',
                'Questa etichetta e cosi forte che vince da sola: anche in posizioni periferiche attira lo sguardo.',
                '#2D6A4F')
    elif punteggio >= 9.0:
        return ('Posizione standard',
                'Etichetta molto persuasiva. Puo essere posizionata in zone standard dello scaffale.',
                '#1A5276')
    elif punteggio >= 8.0:
        return ('Hands level - Zone centrali',
                'Etichetta moderatamente persuasiva. Posizionare ad altezza mano in zone centrali.',
                '#E07A20')
    else:
        return ('Eye level - Centro scaffale',
                'Etichetta poco persuasiva. Necessita della massima visibilita: posizionare eye level al centro.',
                '#C0392B')

def genera_report(img_path, output_path='risultati/'):
    os.makedirs(output_path, exist_ok=True)

    print(f"\nAnalisi di: {img_path}")

    modello_globale, modello_categorie, scaler, metadati = carica_modelli()

    features = estrai_features_immagine(img_path)
    features_scaled = scaler.transform([features])

    punteggio_globale = float(modello_globale.predict(features_scaled)[0])
    punteggio_globale = max(1.0, min(10.0, punteggio_globale))

    punteggi_categorie = modello_categorie.predict(features_scaled)[0]
    punteggi_categorie = [max(1.0, min(10.0, float(p))) for p in punteggi_categorie]

    colonne = metadati['colonne_punteggi']
    classificazione = classifica_etichetta(punteggio_globale)
    posizione, motivazione_scaffale, colore_scaffale = consiglio_scaffale(punteggio_globale)
    descrizione = genera_descrizione_visiva(punteggi_categorie, colonne, punteggio_globale)

    nome_file = os.path.splitext(os.path.basename(img_path))[0]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_path, f'report_{nome_file}_{timestamp}.png')

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor('#F8F3EC')

    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.35)

    ax_img = fig.add_subplot(gs[0, 0])
    img_display = cv2.imread(img_path)
    img_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
    ax_img.imshow(img_display)
    ax_img.axis('off')
    ax_img.set_title('Etichetta Analizzata', fontsize=11, fontweight='bold', color='#3D0A14')

    ax_score = fig.add_subplot(gs[0, 1])
    ax_score.set_xlim(0, 10)
    ax_score.set_ylim(0, 10)
    ax_score.axis('off')

    if punteggio_globale >= 9.5:
        colore_score = '#2D6A4F'
    elif punteggio_globale >= 9.0:
        colore_score = '#1A5276'
    elif punteggio_globale >= 8.0:
        colore_score = '#E07A20'
    else:
        colore_score = '#C0392B'

    cerchio = plt.Circle((5, 5), 3.5, color=colore_score, fill=True, alpha=0.15)
    cerchio2 = plt.Circle((5, 5), 3.5, color=colore_score, fill=False, linewidth=3)
    ax_score.add_patch(cerchio)
    ax_score.add_patch(cerchio2)
    ax_score.text(5, 5.3, f'{punteggio_globale:.2f}', ha='center', va='center',
                  fontsize=32, fontweight='bold', color=colore_score)
    ax_score.text(5, 3.8, '/10', ha='center', va='center', fontsize=14, color='gray')
    ax_score.set_title('Punteggio Globale', fontsize=11, fontweight='bold', color='#3D0A14')

    ax_class = fig.add_subplot(gs[0, 2])
    ax_class.set_xlim(0, 10)
    ax_class.set_ylim(0, 10)
    ax_class.axis('off')
    ax_class.add_patch(plt.Rectangle((0.5, 3.5), 9, 3, color=colore_score, alpha=0.15, linewidth=2))
    ax_class.text(5, 5.5, classificazione, ha='center', va='center',
                  fontsize=11, fontweight='bold', color=colore_score, wrap=True)
    ax_class.text(5, 2.5, 'CLASSIFICAZIONE', ha='center', va='center',
                  fontsize=9, color='gray')
    ax_class.set_title('Valutazione', fontsize=11, fontweight='bold', color='#3D0A14')

    ax_pie = fig.add_subplot(gs[1, 0])
    nomi_brevi = []
    for c in colonne:
        parole = c.split(' ')
        if len(parole) >= 2:
            nomi_brevi.append(parole[0] + '\n' + parole[1])
        else:
            nomi_brevi.append(c[:12])

    colori_torta = ['#6B1A2A', '#9B3A4E', '#C9A84C', '#2D6A4F', '#1A5276'][:len(colonne)]
    wedges, texts, autotexts = ax_pie.pie(
        punteggi_categorie,
        labels=nomi_brevi,
        colors=colori_torta,
        autopct='%1.1f%%',
        startangle=90,
        pctdistance=0.8
    )
    for text in texts:
        text.set_fontsize(7)
    for autotext in autotexts:
        autotext.set_fontsize(7)
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    ax_pie.set_title('Distribuzione Punteggi\nper Categoria', fontsize=11, fontweight='bold', color='#3D0A14')

    ax_bar = fig.add_subplot(gs[1, 1:])
    y_pos = range(len(colonne))
    colori_barre = []
    for p in punteggi_categorie:
        if p >= 9.5:
            colori_barre.append('#2D6A4F')
        elif p >= 9.0:
            colori_barre.append('#1A5276')
        elif p >= 8.0:
            colori_barre.append('#E07A20')
        else:
            colori_barre.append('#C0392B')

    bars = ax_bar.barh(y_pos, punteggi_categorie, color=colori_barre, alpha=0.8, height=0.6)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels([c[:30] for c in colonne], fontsize=8)
    ax_bar.set_xlim(0, 10)
    ax_bar.set_xlabel('Punteggio (0-10)', fontsize=9)
    ax_bar.set_title('Punteggi per Categoria', fontsize=11, fontweight='bold', color='#3D0A14')
    ax_bar.axvline(x=9.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax_bar.text(9.05, len(colonne)-0.3, 'Soglia\nalta', fontsize=7, color='gray')

    for bar, score in zip(bars, punteggi_categorie):
        ax_bar.text(score + 0.1, bar.get_y() + bar.get_height()/2,
                    f'{score:.2f}', va='center', fontsize=9, fontweight='bold')

    ax_scaffale = fig.add_subplot(gs[2, :])
    ax_scaffale.set_xlim(0, 10)
    ax_scaffale.set_ylim(0, 4)
    ax_scaffale.axis('off')
    ax_scaffale.set_facecolor('#F8F3EC')

    ax_scaffale.add_patch(plt.Rectangle((0, 2.2), 10, 1.6, color=colore_scaffale, alpha=0.1))
    ax_scaffale.text(0.3, 3.5, 'POSIZIONAMENTO SCAFFALE CONSIGLIATO:', fontsize=10,
                     fontweight='bold', color='#3D0A14', va='top')
    ax_scaffale.text(0.3, 3.0, f'► {posizione}', fontsize=13,
                     fontweight='bold', color=colore_scaffale, va='top')
    ax_scaffale.text(0.3, 2.3, motivazione_scaffale, fontsize=9,
                     color='#5C3344', va='top', wrap=True,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    ax_scaffale.add_patch(plt.Rectangle((0, 0.3), 10, 1.7, color='#EDE5D8', alpha=0.5))
    ax_scaffale.text(0.3, 1.8, 'DESCRIZIONE PERCETTIVA:', fontsize=9,
                     fontweight='bold', color='#3D0A14', va='top')
    ax_scaffale.text(0.3, 1.3, descrizione, fontsize=10,
                     color='#6B1A2A', va='top', style='italic')

    fig.suptitle('SISTEMA AI VALUTAZIONE ETICHETTE DI VINO — Analisi Neuromarketing',
                 fontsize=14, fontweight='bold', color='#3D0A14', y=0.98)

    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='#F8F3EC')
    plt.close()

    print("\n" + "="*60)
    print("RISULTATI ANALISI")
    print("="*60)
    print(f"Punteggio globale: {punteggio_globale:.2f}/10")
    print(f"Classificazione: {classificazione}")
    print(f"Posizione scaffale: {posizione}")
    print(f"\nPunteggi per categoria:")
    for col, punteggio in zip(colonne, punteggi_categorie):
        print(f"  {col[:40]:40s}: {punteggio:.2f}")
    print(f"\nDescrizione: {descrizione}")
    print(f"\nReport salvato in: {output_file}")
    print("="*60)

    return output_file

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("UTILIZZO: python analisi.py percorso/immagine.jpg")
        print("ESEMPIO:  python analisi.py dataset/immagini/IMG_2757.jpeg")
    else:
        img_path = sys.argv[1]
        if not os.path.exists(img_path):
            print(f"Errore: immagine non trovata in {img_path}")
        else:
            genera_report(img_path)