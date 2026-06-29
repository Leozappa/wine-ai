# ETISCAN – Wine Label AI Evaluation System

Sistema di intelligenza artificiale per la valutazione della persuasività visiva delle etichette di vino, sviluppato come progetto per il corso di Neuromarketing 2025/2026 – Università IULM.

## Descrizione

ETISCAN analizza l'immagine di un'etichetta di vino ed elabora un punteggio globale di persuasività su scala da 1 a 10, insieme a cinque punteggi distinti per eleganza, completezza delle informazioni, coerenza cromatica, qualità del design e attrattività per i giovani. Il sistema genera un report visivo scaricabile in formato PNG con grafici e consiglio di posizionamento a scaffale.

## Requisiti

Assicurarsi di avere Python 3.11 installato. Le dipendenze si installano con:

bash
pip install -r requirements.txt


## Utilizzo

1. Eseguire la preparazione dei dati:
bash
python preparazione_dati.py

2. Addestrare il modello:
bash
python addestramento.py

3. Avviare l'applicazione web:
bash
python app.py

4. Aprire il browser su http://localhost:5000, caricare un'immagine di etichetta e ottenere il report.

## Demo online

Il prototipo è pubblicato su Railway ed è raggiungibile da qualsiasi dispositivo senza installazioni.

## Stack tecnologico

- Python 3.11
- Flask
- scikit-learn (GradientBoostingRegressor, RandomForestRegressor)
- OpenCV, Pillow
- Pandas, NumPy, Matplotlib, Seaborn

## Autori

Agnese Delucchi · Leonardo Zappavigna · Delvina Vartic
