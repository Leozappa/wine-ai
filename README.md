# ETISCAN – Sistema AI per la valutazione delle etichette di vino

> Progetto di Machine Learning per la valutazione automatica della persuasività visiva delle etichette di vino. Il sistema predice un punteggio globale e cinque punteggi di categoria attraverso una pipeline di estrazione di feature visive e modelli ensemble, con interfaccia web e deploy online.

## Obiettivo del progetto

L'obiettivo principale di questo progetto è sviluppare un sistema predittivo in grado di stimare la *persuasività visiva* di un'etichetta di vino su scala da 1 a 10, insieme a cinque dimensioni valutative distinte:

1. *Eleganza:* qualità estetica complessiva dell'etichetta
2. *Completezza:* presenza e chiarezza delle informazioni riportate
3. *Coerenza cromatica:* armonia tra i colori dell'etichetta e il prodotto
4. *Qualità del design:* livello di cura grafica e compositiva
5. *Attrattività per i giovani:* capacità di attrarre un pubblico under 30

## Dettagli implementativi

Il progetto è strutturato in quattro moduli Python che operano in sequenza, coordinati da un'applicazione web Flask.

### Estrazione delle feature visive

Ogni immagine viene trasformata in un vettore di *27 feature numeriche* tramite OpenCV:

- *Colori medi RGB:* media dei tre canali (3 valori)
- *Deviazione standard RGB:* variabilità cromatica per canale (3 valori)
- *Contrasto:* deviazione standard dell'immagine in scala di grigi (1 valore)
- *Luminosità:* media dell'intensità luminosa complessiva (1 valore)
- *Nitidezza:* varianza del filtro Laplaciano (1 valore)
- *Istogrammi di colore:* distribuzione normalizzata con 6 classi per canale (18 valori)

### Modelli di machine learning

Vengono utilizzati due modelli distinti per due compiti differenti:

| Modello | Tipo | Utilizzo |
|---|---|---|
| *GradientBoostingRegressor* | Ensemble sequenziale | Predizione del punteggio globale di persuasività |
| *MultiOutputRegressor + RandomForestRegressor* | Ensemble parallelo multi-output | Predizione simultanea dei cinque punteggi di categoria |

Il GradientBoostingRegressor è configurato con 200 stimatori, learning rate 0.05 e profondità massima 4, parametri scelti per contenere il rischio di overfitting su un dataset di piccole dimensioni. Il RandomForestRegressor con 200 alberi è racchiuso in un MultiOutputRegressor che addestra un regressore dedicato per ciascun target.

## Dataset

| Parametro | Dettaglio |
|---|---|
| *Numero di campioni* | 43 etichette di vino reali |
| *Fonte* | Sessioni del Laboratorio di Neuromarketing – Università IULM |
| *Etichette* | Punteggi assegnati da un panel di esperti su 5 dimensioni |
| *Formato* | File Excel (dataset_laboratorio.xlsx) + immagini JPG |
| *Split* | 80% training / 20% test |

## Pipeline

preparazione_dati.py  →  addestramento.py  →  analisi.py

↓

app.py (Flask)

1. preparazione_dati.py – legge il dataset Excel, carica le immagini ed estrae le 27 feature visive, salvando i dati in formato NumPy
2. addestramento.py – addestra i due modelli e li salva su disco
3. analisi.py – ricarica i modelli, analizza una nuova immagine e genera il report PNG
4. app.py – espone l'intera pipeline tramite interfaccia web con tre endpoint REST

## Per iniziare

### Prerequisiti

Assicurarsi di avere Python 3.11 installato e di utilizzare un ambiente virtuale (es. Conda) per isolare le dipendenze.

### Installazione

1. *Clonare il repository:*
git clone https://github.com/Leozappa/wine-ai.git

2. *Navigare nella cartella del progetto:*
cd wine-ai

3. *Installare le librerie richieste:*
pip install -r requirements.txt

4. *Attivare l'ambiente virtuale:*
conda activate [nome_ambiente]

## Esecuzione

Eseguire i moduli in sequenza dal terminale:

python preparazione_dati.py
python addestramento.py
python app.py

Aprire il browser su http://localhost:5000, caricare un'immagine di etichetta e ottenere il report visivo in formato PNG.

## Demo online

Il prototipo è pubblicato su *Railway* ed è raggiungibile da qualsiasi dispositivo senza installazioni locali.

## Stack tecnologico

| Componente | Tecnologia |
|---|---|
| Linguaggio | Python 3.11 |
| Framework web | Flask |
| Machine learning | scikit-learn (GradientBoostingRegressor, RandomForestRegressor, MultiOutputRegressor) |
| Computer vision | OpenCV (cv2) |
| Elaborazione immagini | Pillow (PIL) |
| Analisi dati | Pandas, NumPy |
| Visualizzazione | Matplotlib, Seaborn |
| Hosting | Railway |

## Autori

Agnese Delucchi · Leonardo Zappavigna · Delvina Vartic

Università IULM – Data Mining & Text Analytics 2025/2026
