import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau


class LSTMModel(nn.Module):
    def __init__(self, input_size=7, hidden_size=128, num_layers=2, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


def _coerce_float_frame(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if df[c].dtype == 'object':
            try:
                df[c] = df[c].str.replace(',', '.').astype(float)
            except Exception:
                df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def _sliding_sequences(X: np.ndarray, y: np.ndarray, time_steps: int):
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i:i + time_steps])
        y_seq.append(y[i + time_steps])
    return np.array(X_seq), np.array(y_seq)


def _print_epoch(epoch: int, train_loss: float, val_loss: float):
    print(f"[Epoch {epoch}/100] Train Loss: {train_loss/10:.4f} | Val Loss: {val_loss/10:.4f}")


def _split_train_val_test(X_seq: np.ndarray, y_seq: np.ndarray):
    X_train, X_temp, y_train, y_temp = train_test_split(X_seq, y_seq, test_size=0.3, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)
    return X_train, X_val, X_test, y_train, y_val, y_test


def _to_tensors(*arrays):
    out = []
    for a in arrays:
        out.append(torch.FloatTensor(a))
    return out


def _fit_standard_scalers(df: pd.DataFrame, features: list):
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X = scaler_X.fit_transform(df[features])
    y = scaler_y.fit_transform(df['Temperature'].values.reshape(-1, 1))
    return scaler_X, scaler_y, X, y


def _select_and_rename(df: pd.DataFrame) -> pd.DataFrame:
    selected_columns = ['CO(GT)', 'C6H6(GT)', 'NO2(GT)', 'NOx(GT)', 'RH', 'AH', 'T']
    df = df[selected_columns]
    df.columns = ['CO', 'C6H6', 'NO2', 'NOx', 'RH', 'AH', 'Temperature']
    return df


def _read_and_prepare(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path, sep=';', decimal=',')
    df = df.dropna(axis=1, how='all').dropna()
    df = _select_and_rename(df)
    df = _coerce_float_frame(df)
    return df


def train_model(data_path="AirQualityUCI.csv"):
    df = _read_and_prepare(data_path)
    features = ['CO', 'C6H6', 'NO2', 'NOx', 'RH', 'AH', 'Temperature']
    scaler_X, scaler_y, X, y = _fit_standard_scalers(df, features)
    time_steps = 24
    X_seq, y_seq = _sliding_sequences(X, y, time_steps)
    X_train, X_val, X_test, y_train, y_val, y_test = _split_train_val_test(X_seq, y_seq)
    X_train, y_train, X_val, y_val, X_test, y_test = _to_tensors(X_train, y_train, X_val, y_val, X_test, y_test)
    model = LSTMModel(input_size=X_train.shape[2])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    best_val_loss = float('inf')
    best_model_state = None
    for epoch in range(1, 101):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_output = model(X_val)
            val_loss = criterion(val_output, y_val)
        scheduler.step(val_loss)
        _print_epoch(epoch, loss.item(), val_loss.item())
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_model_state = model.state_dict()
    torch.save(best_model_state, 'model_weights.pt')
    joblib.dump(scaler_X, 'scaler_X.pkl')
    joblib.dump(scaler_y, 'scaler_y.pkl')
    torch.save(X_test, 'X_test.pt')
    torch.save(y_test, 'y_test.pt')
    print("训练完成，模型和归一化器已保存。")


if __name__ == "__main__":
    train_model()
