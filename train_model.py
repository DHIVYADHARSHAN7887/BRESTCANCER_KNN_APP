import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os

# ─── Load Dataset ───────────────────────────────────────────────
df = pd.read_csv("KNNAlgorithmDataset_csv.xls")

# ─── Clean Data ─────────────────────────────────────────────────
df.drop(columns=["id", "Unnamed: 32"], inplace=True, errors="ignore")
df.dropna(inplace=True)

# ─── Encode Target ──────────────────────────────────────────────
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

# ─── Features & Target ──────────────────────────────────────────
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

feature_names = X.columns.tolist()

# ─── Train / Test Split ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─── Scale Features ─────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ─── Find Best K ────────────────────────────────────────────────
best_k, best_acc = 1, 0
for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    acc = accuracy_score(y_test, knn.predict(X_test_scaled))
    if acc > best_acc:
        best_acc, best_k = acc, k

print(f"Best K: {best_k}  |  Accuracy: {best_acc * 100:.2f}%")

# ─── Train Final Model ──────────────────────────────────────────
model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Benign", "Malignant"]))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ─── Save Artifacts ─────────────────────────────────────────────
with open("knn_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("feature_names.pkl", "wb") as f:
    pickle.dump(feature_names, f)

print("\n✅ Saved: knn_model.pkl, scaler.pkl, feature_names.pkl")
