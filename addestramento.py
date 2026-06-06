import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

MODELLO_PATH = 'modello/'

print("Caricamento dati...")
X = np.load(os.path.join(MODELLO_PATH, 'X.npy'))
y = np.load(os.path.join(MODELLO_PATH, 'y.npy'))
y_categorie = np.load(os.path.join(MODELLO_PATH, 'y_categorie.npy'))

with open(os.path.join(MODELLO_PATH, 'metadati.pkl'), 'rb') as f:
    metadati = pickle.load(f)

print(f"Dataset: {X.shape[0]} campioni, {X.shape[1]} features")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

if X.shape[0] >= 5:
    X_train, X_test, y_train, y_test, yc_train, yc_test = train_test_split(
        X_scaled, y, y_categorie, test_size=0.2, random_state=42
    )
else:
    X_train, X_test = X_scaled, X_scaled
    y_train, y_test = y, y
    yc_train, yc_test = y_categorie, y_categorie

print("\nAddestramento modello punteggio globale...")
modello_globale = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    random_state=42
)
modello_globale.fit(X_train, y_train)
y_pred = modello_globale.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MAE punteggio globale: {mae:.3f}")
print(f"R2 punteggio globale: {r2:.3f}")

print("\nAddestramento modello categorie...")
modello_categorie = MultiOutputRegressor(
    RandomForestRegressor(n_estimators=200, random_state=42)
)
modello_categorie.fit(X_train, yc_train)
yc_pred = modello_categorie.predict(X_test)
mae_cat = mean_absolute_error(yc_test, yc_pred)
print(f"MAE categorie: {mae_cat:.3f}")

print("\nSalvataggio modelli...")
with open(os.path.join(MODELLO_PATH, 'modello_globale.pkl'), 'wb') as f:
    pickle.dump(modello_globale, f)

with open(os.path.join(MODELLO_PATH, 'modello_categorie.pkl'), 'wb') as f:
    pickle.dump(modello_categorie, f)

with open(os.path.join(MODELLO_PATH, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)

print("Modelli salvati!")

print("\nGenerazione grafico performance...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(y_test, y_pred, alpha=0.7, color='darkred', s=80)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
axes[0].set_xlabel('Punteggio Reale')
axes[0].set_ylabel('Punteggio Predetto')
axes[0].set_title('Performance Modello Globale')
axes[0].text(0.05, 0.95, f'R2={r2:.3f}\nMAE={mae:.3f}',
             transform=axes[0].transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

colonne = metadati['colonne_punteggi']
nomi_brevi = [c.split(' ')[0] for c in colonne]
mae_per_categoria = [mean_absolute_error(yc_test[:, i], yc_pred[:, i]) for i in range(len(colonne))]

axes[1].bar(nomi_brevi, mae_per_categoria, color='darkred', alpha=0.7)
axes[1].set_title('Errore per Categoria')
axes[1].set_ylabel('MAE')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(MODELLO_PATH, 'performance.png'), dpi=150, bbox_inches='tight')
plt.close()

print("Grafico performance salvato in modello/performance.png")
print("\n=== ADDESTRAMENTO COMPLETATO ===")
print(f"Precisione modello: {100 - mae*10:.1f}%")