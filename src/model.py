from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn


MODEL_DIR = Path(__file__).resolve().parent.parent / "model"


class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        output_size,
        target_length=8,
    ):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.target_length = target_length
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
        )
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        hidden_cell = (
            torch.zeros(self.num_layers, x.size(0), self.hidden_size),
            torch.zeros(self.num_layers, x.size(0), self.hidden_size),
        )
        out, _ = self.lstm(x, hidden_cell)
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out.reshape(x.size(0), self.target_length, 2)


class TargetPredictor:
    def __init__(
        self,
        model_path=MODEL_DIR / "lstm_model.pth",
        x_scaler_path=MODEL_DIR / "x_scaler.pkl",
        y_scaler_path=MODEL_DIR / "y_scaler.pkl",
        input_size=4,
        hidden_size=64,
        num_layers=2,
        output_size=16,
        target_length=8,
    ):
        self.target_length = target_length
        self.x_scaler = joblib.load(x_scaler_path)
        self.y_scaler = joblib.load(y_scaler_path)
        self.model = LSTMModel(
            input_size,
            hidden_size,
            num_layers,
            output_size,
            target_length,
        )
        self.model.load_state_dict(
            torch.load(model_path, map_location=torch.device("cpu"))
        )
        self.model.eval()

    def to_global_acceleration(self, relative_acceleration, current_vel):
        heading = np.arctan2(current_vel[1], current_vel[0])
        rotation = np.array(
            [
                [np.cos(heading), np.sin(heading)],
                [-np.sin(heading), np.cos(heading)],
            ]
        )
        return relative_acceleration @ rotation

    def predict(self, history):
        history = np.asarray(history, dtype=float)
        current_pos = history[-1, 0:2]
        current_vel = history[-1, 2:4]
        heading = np.arctan2(current_vel[1], current_vel[0])
        rotation = np.array(
            [
                [np.cos(heading), np.sin(heading)],
                [-np.sin(heading), np.cos(heading)],
            ]
        )
        delta_pos = history[:, 0:2] - current_pos
        relative_pos = delta_pos @ rotation.T
        relative_vel = history[:, 2:4] @ rotation.T
        sequence = np.hstack((relative_pos, relative_vel))
        scaled_sequence = self.x_scaler.transform(sequence).reshape(1, 40, 4)
        sequence_tensor = torch.tensor(scaled_sequence, dtype=torch.float32)
        with torch.no_grad():
            scaled_prediction = self.model(sequence_tensor)
        relative_acceleration = self.y_scaler.inverse_transform(
            scaled_prediction.numpy().reshape(-1, 2)
        ).reshape(self.target_length, 2)
        return self.to_global_acceleration(
            relative_acceleration,
            current_vel,
        )
