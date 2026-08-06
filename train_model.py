import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Chemins
train_path = os.path.join("DataSet", "train.csv")
model_dir = "Model"

# Lecture du CSV avec gestion d'encodage
print("Chargement des données...")
encodings = ['utf-8', 'latin1', 'cp1252', 'ISO-8859-1']
train = None
for enc in encodings:
    try:
        train = pd.read_csv(train_path, encoding=enc)
        print(f"Encodage utilisé : {enc}")
        break
    except UnicodeDecodeError:
        continue
if train is None:
    raise Exception("Impossible de lire le CSV.")

# Afficher les colonnes disponibles
print("Colonnes disponibles :", train.columns.tolist())

# --- NETTOYAGE DES DONNÉES ---
# 1. Supprimer les lignes où 'text' est manquant (NaN)
initial_shape = train.shape
train = train.dropna(subset=['text'])
print(f"Lignes après suppression des NaN dans 'text' : {train.shape[0]} (était {initial_shape[0]})")

# 2. Optionnel : supprimer les lignes où 'text' est une chaîne vide
train = train[train['text'].str.strip() != '']
print(f"Lignes après suppression des textes vides : {train.shape[0]}")

# 3. Vérifier que 'sentiment' ne contient pas de NaN (si c'est le cas, on les supprime aussi)
if train['sentiment'].isnull().any():
    train = train.dropna(subset=['sentiment'])
    print("Des NaN dans 'sentiment' ont été supprimés.")

# 4. Afficher la répartition des classes
print("Répartition des sentiments :")
print(train['sentiment'].value_counts())

# Sélection des colonnes
X_train = train["text"]
y_train = train["sentiment"]

# Vectorisation TF-IDF
print("Vectorisation TF-IDF...")
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)

# Entraînement du modèle
print("Entraînement du modèle...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Sauvegarde
print("Sauvegarde dans Model/...")
os.makedirs(model_dir, exist_ok=True)
joblib.dump(model, os.path.join(model_dir, "sentiment_analysis_model.pkl"))
joblib.dump(vectorizer, os.path.join(model_dir, "tfidf_vectorizer.pkl"))

print("✅ Fichiers générés avec succès :")
print("   - Model/sentiment_analysis_model.pkl")
print("   - Model/tfidf_vectorizer.pkl")