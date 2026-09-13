from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from training.utils import cargar_datos, evaluar_y_reportar


# 1. Definición de la Arquitectura CNN 1D en PyTorch
class CNN1D(nn.Module):

    def __init__(self, num_features):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=1, out_channels=32, kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        # x shape: (batch_size, 1, num_features)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)  # (batch_size, 64, 1)
        x = x.view(x.size(0), -1)  # Flatten: (batch_size, 64)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x


def calcular_permutation_importance_pytorch(
    modelo, X_val_scaled, y_val, feature_names, device, n_repeats=5
):
    print("🔍 Calculando Permutation Importance para CNN 1D (PyTorch)...")
    modelo.eval()
    y_val_arr = np.array(y_val)

    X_tensor = (
        torch.tensor(X_val_scaled, dtype=torch.float32).unsqueeze(1).to(device)
    )
    with torch.no_grad():
        preds_base = modelo(X_tensor).cpu().numpy().flatten()
    acc_base = np.mean((preds_base >= 0.5).astype(int) == y_val_arr)

    importances = []
    for col_idx in range(X_val_scaled.shape[1]):
        drop_accs = []
        for _ in range(n_repeats):
            X_permuted = X_val_scaled.copy()
            X_permuted[:, col_idx] = np.random.permutation(
                X_permuted[:, col_idx]
            )
            X_perm_tensor = (
                torch.tensor(X_permuted, dtype=torch.float32)
                .unsqueeze(1)
                .to(device)
            )

            with torch.no_grad():
                preds_perm = (
                    modelo(X_perm_tensor).cpu().numpy().flatten()
                )
            acc_perm = np.mean((preds_perm >= 0.5).astype(int) == y_val_arr)
            drop_accs.append(acc_base - acc_perm)

        importances.append(np.mean(drop_accs))

    return pd.Series(importances, index=feature_names)


def entrenar_cnn():
    datos = cargar_datos()

    root = Path(__file__).resolve().parent.parent
    dir_models = root / "models"
    dir_models.mkdir(parents=True, exist_ok=True)

    # Preprocesamiento
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_tr_clean = imputer.fit_transform(datos["X_tr"])
    X_tr_scaled = scaler.fit_transform(X_tr_clean)

    X_es_clean = imputer.transform(datos["X_es"])
    X_es_scaled = scaler.transform(X_es_clean)

    X_val_clean = imputer.transform(datos["X_val"])
    X_val_scaled = scaler.transform(X_val_clean)

    # Dispositivo (CPU / GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Datasets de PyTorch (forma: N, Channels=1, Features)
    train_ds = TensorDataset(
        torch.tensor(X_tr_scaled, dtype=torch.float32).unsqueeze(1),
        torch.tensor(datos["y_tr"].values, dtype=torch.float32).unsqueeze(1),
    )
    eval_ds = TensorDataset(
        torch.tensor(X_es_scaled, dtype=torch.float32).unsqueeze(1),
        torch.tensor(datos["y_es"].values, dtype=torch.float32).unsqueeze(1),
    )

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    # Crear Modelo
    num_features = datos["X_tr"].shape[1]
    modelo = CNN1D(num_features).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(modelo.parameters(), lr=0.001)

    print("🚀 Entrenando 1D CNN con PyTorch...")

    # Entrenamiento con Early Stopping básico
    best_loss = float("inf")
    patience, patience_counter = 15, 0

    for epoch in range(100):
        modelo.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            out = modelo(X_b)
            loss = criterion(out, y_b)
            loss.backward()
            optimizer.step()

        # Validación
        modelo.eval()
        X_es_t = (
            torch.tensor(X_es_scaled, dtype=torch.float32)
            .unsqueeze(1)
            .to(device)
        )
        y_es_t = (
            torch.tensor(datos["y_es"].values, dtype=torch.float32)
            .unsqueeze(1)
            .to(device)
        )

        with torch.no_grad():
            val_out = modelo(X_es_t)
            val_loss = criterion(val_out, y_es_t).item()

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(
                modelo.state_dict(), dir_models / "best_cnn_weights.pt"
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping en época {epoch + 1}")
                break

    # Cargar los mejores pesos
    modelo.load_state_dict(torch.load(dir_models / "best_cnn_weights.pt"))
    modelo.eval()

    # Predicción en Validación
    X_val_t = (
        torch.tensor(X_val_scaled, dtype=torch.float32).unsqueeze(1).to(device)
    )
    with torch.no_grad():
        y_probs_val = modelo(X_val_t).cpu().numpy().flatten()
    y_pred_val = (y_probs_val >= 0.5).astype(int)

    # Evaluar y guardar gráficas en docs/cnn/
    dir_out = evaluar_y_reportar(
        "CNN", datos["y_val"], y_pred_val, datos["val_df"]
    )

    # Permutation Importance
    importancias = calcular_permutation_importance_pytorch(
        modelo, X_val_scaled, datos["y_val"], datos["X_train"].columns, device
    )
    top15 = importancias.nlargest(15).sort_values(ascending=True)

    ruta_grafica = dir_out / "importancia.png"
    plt.figure(figsize=(10, 8))
    top15.plot(kind="barh", color="#d32f2f")
    plt.title("CNN 1D (PyTorch) - Top 15 Métricas por Importancia")
    plt.xlabel("Caída Promedio en Accuracy al Permutar Variable")
    plt.tight_layout()
    plt.savefig(ruta_grafica, dpi=150)
    plt.close()
    print(f"📉 Gráfica de importancia guardada en: {ruta_grafica}")

    # Guardar Preprocesadores
    ruta_prep = dir_models / "preprocesadores_cnn_altur.joblib"
    joblib.dump({"imputer": imputer, "scaler": scaler}, ruta_prep)

    print(f"💾 Modelo CNN guardado en: {dir_models / 'best_cnn_weights.pt'}")
    print(f"💾 Preprocesadores guardados en: {ruta_prep}")


if __name__ == "__main__":
    entrenar_cnn()