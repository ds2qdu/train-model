# ============================================
# US Stock Prediction E2E Prototype
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

# ============================================
# 1. Data Collection
# ============================================
print("=" * 50)
print("1. Data Collection")
print("=" * 50)

# S&P 500 Index stock data (2 years)
ticker = "^GSPC"
end_date = datetime.now()
start_date = end_date - timedelta(days=3600)

df = yf.download(ticker, start=start_date, end=end_date)
print(f"Collected data: {len(df)} days")
print(f"Period: {df.index[0].date()} ~ {df.index[-1].date()}")
print(f"\nLast 5 days data:")
print(df.tail())

# Data validation: Charts
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(df['Close'])
plt.title(f'{ticker} Close Price')
plt.xlabel('Date')
plt.ylabel('Price ($)')

plt.subplot(1, 2, 2)
plt.fill_between(df.index, df['Volume'].values.flatten(), alpha=0.7)  # width parameter removed
plt.title(f'{ticker} Volume')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.tight_layout()
plt.savefig('data_check.png')
plt.show()

print("✅ Data collection complete - validated with charts")

# ============================================
# 2. Data Preprocessing
# ============================================
print("\n" + "=" * 50)
print("2. Data Preprocessing")
print("=" * 50)

# Use Close price only
data = df['Close'].values.reshape(-1, 1)

# Normalization
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Create sequences (30 days → 5 days prediction)
def create_sequences(data, seq_length=30, pred_length=5):
    X, y = [], []
    for i in range(len(data) - seq_length - pred_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length:i+seq_length+pred_length])
    return np.array(X), np.array(y)

SEQ_LENGTH = 30
PRED_LENGTH = 5

X, y = create_sequences(data_scaled, SEQ_LENGTH, PRED_LENGTH)
print(f"Input shape: {X.shape}  (samples, 30 days, 1)")
print(f"Output shape: {y.shape}  (samples, 5 days, 1)")

# Train/Test split (80/20)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"Training data: {len(X_train)}")
print(f"Test data: {len(X_test)}")

# Convert to PyTorch tensors
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train).squeeze(-1)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.FloatTensor(y_test).squeeze(-1)

train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t), 
    batch_size=32, shuffle=True
)

print("✅ Preprocessing complete")

# ============================================
# 3. Model Definition (LSTM)
# ============================================
print("\n" + "=" * 50)
print("3. Model Definition")
print("=" * 50)

class StockLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, pred_length=5):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        self.fc = nn.Linear(hidden_size, pred_length)
    
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # lstm_out[:, -1, :]: use last timestep only
        out = self.fc(lstm_out[:, -1, :])
        return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

model = StockLSTM(pred_length=PRED_LENGTH).to(device)
print(model)
print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================
# 4. Model Training
# ============================================
print("\n" + "=" * 50)
print("4. Model Training")
print("=" * 50)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 50
train_losses = []

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.6f}")

# Training validation: Loss graph
plt.figure(figsize=(10, 4))
plt.plot(train_losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.savefig('training_loss.png')
plt.show()

print("✅ Training complete")

# ============================================
# 5. Prediction Test
# ============================================
print("\n" + "=" * 50)
print("5. Prediction Test")
print("=" * 50)

model.eval()
with torch.no_grad():
    X_test_device = X_test_t.to(device)
    predictions = model(X_test_device).cpu().numpy()

# Inverse normalization
predictions_inv = scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(-1, PRED_LENGTH)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1, PRED_LENGTH)

# Prediction validation: Actual vs Predicted comparison
plt.figure(figsize=(14, 6))

# Compare last 100 days
n_show = min(100, len(predictions_inv))
for i in range(0, n_show, 10):
    actual = y_test_inv[i]
    pred = predictions_inv[i]
    
    x_actual = range(i, i + PRED_LENGTH)
    plt.plot(x_actual, actual, 'b-', alpha=0.5, label='Actual' if i == 0 else '')
    plt.plot(x_actual, pred, 'r--', alpha=0.5, label='Predicted' if i == 0 else '')

plt.title(f'{ticker} Actual vs Predicted (5-day prediction)')
plt.xlabel('Sample Index')
plt.ylabel('Price ($)')
plt.legend()
plt.savefig('prediction_result.png')
plt.show()

# Calculate MAPE
mape = np.mean(np.abs((y_test_inv - predictions_inv) / y_test_inv)) * 100
print(f"MAPE (Mean Absolute Percentage Error): {mape:.2f}%")

# ============================================
# 6. Future Prediction with Latest Data
# ============================================
print("\n" + "=" * 50)
print("6. Future 5-day Prediction")
print("=" * 50)

# Predict next 5 days using last 30 days
last_30_days = data_scaled[-SEQ_LENGTH:]
last_30_days_t = torch.FloatTensor(last_30_days).unsqueeze(0).to(device)

model.eval()
with torch.no_grad():
    future_pred = model(last_30_days_t).cpu().numpy()

future_pred_inv = scaler.inverse_transform(future_pred.reshape(-1, 1)).flatten()

print(f"\nCurrent close price: ${data[-1][0]:.2f}")
print(f"\nNext 5 days prediction:")
for i, pred in enumerate(future_pred_inv):
    change = ((pred - data[-1][0]) / data[-1][0]) * 100
    arrow = "📈" if change > 0 else "📉"
    print(f"  Day {i+1}: ${pred:.2f} ({change:+.2f}%) {arrow}")

# Visualization
plt.figure(figsize=(12, 5))
recent_prices = scaler.inverse_transform(data_scaled[-60:]).flatten()
x_recent = range(60)
x_future = range(59, 59 + PRED_LENGTH + 1)

plt.plot(x_recent, recent_prices, 'b-', label='Last 60 days')
plt.plot(x_future, [recent_prices[-1]] + list(future_pred_inv), 'r--', marker='o', label='5-day prediction')
plt.axvline(x=59, color='gray', linestyle=':', alpha=0.5)
plt.title(f'{ticker} Future 5-day Prediction')
plt.xlabel('Days')
plt.ylabel('Price ($)')
plt.legend()
plt.savefig('future_prediction.png')
plt.show()

print("\n" + "=" * 50)
print("✅ E2E Prototype Complete!")
print("=" * 50)
