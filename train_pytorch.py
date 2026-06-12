import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, roc_auc_score

# ── LOAD & PREPROCESS ────────────────────────────────
df = pd.read_csv('data/creditcard.csv')
scaler = StandardScaler()
df[['Time', 'Amount']] = scaler.fit_transform(df[['Time', 'Amount']])
X = df.drop('Class', axis=1).values
y = df['Class'].values

# ── SMOTE ────────────────────────────────────────────
X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)

# ── SPLIT & CONVERT ──────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test  = torch.tensor(X_test,  dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test  = torch.tensor(y_test,  dtype=torch.float32).unsqueeze(1)

# ── DATALOADER ───────────────────────────────────────
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=256, shuffle=True)

# ── MODEL ────────────────────────────────────────────
class FraudNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.network(x)

model = FraudNet(input_size=X_train.shape[1])

# ── TRAINING ─────────────────────────────────────────
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

for epoch in range(20):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        p = model(X_batch)
        loss = criterion(p, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()

    if epoch % 5 == 0:
        model.eval()
        with torch.no_grad():
            test_loss = criterion(model(X_test), y_test)
        print(f"Epoch {epoch} | Train Loss: {total_loss/len(train_loader):.4f} | Test Loss: {test_loss.item():.4f}")

# ── EVALUATION ───────────────────────────────────────
model.eval()
with torch.no_grad():
    preds = (model(X_test) > 0.5).numpy().astype(int)
    probs = model(X_test).numpy()

print("\nClassification Report:")
print(classification_report(y_test.numpy(), preds))
print("ROC-AUC Score:", roc_auc_score(y_test.numpy(), probs))