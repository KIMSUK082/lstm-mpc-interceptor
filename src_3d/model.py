from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn


MODEL_ROOT = Path(__file__).resolve().parents[1] / "model" / "3d"
MODEL_DIR = MODEL_ROOT / "acceleration"
POSITION_MODEL_DIR = MODEL_ROOT / "position"


def local_basis(velocity):
    velocity = np.asarray(velocity, dtype=float)
    speed = max(np.linalg.norm(velocity), 1e-8)

    forward = velocity / speed
    world_up = np.array([0.0, 0.0, 1.0])

    side = np.cross(world_up, forward)
    side_norm = np.linalg.norm(side)

    if side_norm < 1e-8:
        reference = np.array([0.0, 1.0, 0.0])
        side = np.cross(reference, forward)
        side_norm = np.linalg.norm(side)

    side = side / max(side_norm, 1e-8)
    vertical = np.cross(forward, side)

    return np.vstack(
        (
            forward,
            vertical,
            side,
        )
    )


class LSTMModel(nn.Module):

    def __init__(
        self,
        input_size=6,
        hidden_size=64,
        num_layers=2,
        output_size=24,
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

        self.fc1 = nn.Linear(
            hidden_size,
            hidden_size // 2,
        )
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(
            hidden_size // 2,
            output_size,
        )

    def forward(self, x):
        hidden_state = torch.zeros(
            self.num_layers,
            x.size(0),
            self.hidden_size,
            device=x.device,
        )

        cell_state = torch.zeros(
            self.num_layers,
            x.size(0),
            self.hidden_size,
            device=x.device,
        )

        output, _ = self.lstm(
            x,
            (hidden_state, cell_state),
        )

        output = output[:, -1, :]
        output = self.fc1(output)
        output = self.relu(output)
        output = self.fc2(output)

        return output.reshape(
            x.size(0),
            self.target_length,
            3,
        )


class TargetPredictor:
    """Predict eight local acceleration commands for method A."""

    def __init__(
        self,
        model_path=MODEL_DIR / "lstm_model.pth",
        x_scaler_path=MODEL_DIR / "x_scaler.pkl",
        y_scaler_path=MODEL_DIR / "y_scaler.pkl",
        target_length=8,
    ):
        self.target_length = target_length

        self.x_scaler = joblib.load(x_scaler_path)
        self.y_scaler = joblib.load(y_scaler_path)

        self.model = LSTMModel(
            input_size=6,
            hidden_size=64,
            num_layers=2,
            output_size=target_length * 3,
            target_length=target_length,
        )

        self.model.load_state_dict(
            torch.load(
                model_path,
                map_location=torch.device("cpu"),
            )
        )
        self.model.eval()

    def predict(self, history):
        history = np.asarray(history, dtype=float)

        if history.shape != (40, 6):
            raise ValueError(
                "history must have shape (40, 6)."
            )

        current_position = history[-1, 0:3]
        current_velocity = history[-1, 3:6]

        rotation = local_basis(current_velocity)

        delta_position = history[:, 0:3] - current_position
        relative_position = delta_position @ rotation.T
        relative_velocity = history[:, 3:6] @ rotation.T

        sequence = np.hstack(
            (
                relative_position,
                relative_velocity,
            )
        )

        scaled_sequence = self.x_scaler.transform(
            sequence
        ).reshape(1, 40, 6)

        sequence_tensor = torch.tensor(
            scaled_sequence,
            dtype=torch.float32,
        )

        with torch.no_grad():
            scaled_prediction = self.model(sequence_tensor)

        local_acceleration = self.y_scaler.inverse_transform(
            scaled_prediction.numpy().reshape(-1, 3)
        ).reshape(self.target_length, 3)

        return local_acceleration


class PositionLSTMModel(nn.Module):
    """LSTM used by method B to predict future relative positions."""

    def __init__(
        self,
        input_size=6,
        hidden_size=96,
        num_layers=2,
        target_length=100,
    ):
        super(PositionLSTMModel, self).__init__()

        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.target_length = int(target_length)

        self.lstm = nn.LSTM(
            input_size,
            self.hidden_size,
            self.num_layers,
            batch_first=True,
        )

        self.fc1 = nn.Linear(
            self.hidden_size,
            self.hidden_size // 2,
        )
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(
            self.hidden_size // 2,
            self.target_length * 3,
        )

    def forward(self, sequence):
        hidden_state = sequence.new_zeros(
            self.num_layers,
            sequence.size(0),
            self.hidden_size,
        )

        cell_state = sequence.new_zeros(
            self.num_layers,
            sequence.size(0),
            self.hidden_size,
        )

        output, _ = self.lstm(
            sequence,
            (hidden_state, cell_state),
        )

        output = output[:, -1, :]
        output = self.fc1(output)
        output = self.relu(output)
        output = self.fc2(output)

        return output.reshape(
            sequence.size(0),
            self.target_length,
            3,
        )


class PositionTargetPredictor:
    """Predict 100 absolute future positions for method B."""

    def __init__(
        self,
        model_path=POSITION_MODEL_DIR / "lstm_model.pth",
        x_scaler_path=POSITION_MODEL_DIR / "x_scaler.pkl",
        y_scaler_path=POSITION_MODEL_DIR / "y_scaler.pkl",
        history_length=40,
        target_length=100,
        hidden_size=96,
        num_layers=2,
    ):
        self.history_length = int(history_length)
        self.target_length = int(target_length)

        self.x_scaler = joblib.load(x_scaler_path)
        self.y_scaler = joblib.load(y_scaler_path)

        self.model = PositionLSTMModel(
            input_size=6,
            hidden_size=hidden_size,
            num_layers=num_layers,
            target_length=self.target_length,
        )

        self.model.load_state_dict(
            torch.load(
                model_path,
                map_location=torch.device("cpu"),
            )
        )
        self.model.eval()

    def predict(self, history):
        history = np.asarray(history, dtype=float)
        expected_shape = (self.history_length, 6)

        if history.shape != expected_shape:
            raise ValueError(
                f"history must have shape {expected_shape}."
            )

        current_position = history[-1, 0:3]
        current_velocity = history[-1, 3:6]
        rotation = local_basis(current_velocity)

        delta_position = history[:, 0:3] - current_position
        relative_position = delta_position @ rotation.T
        relative_velocity = history[:, 3:6] @ rotation.T

        sequence = np.hstack(
            (
                relative_position,
                relative_velocity,
            )
        )

        scaled_sequence = self.x_scaler.transform(
            sequence
        ).reshape(1, self.history_length, 6)

        sequence_tensor = torch.tensor(
            scaled_sequence,
            dtype=torch.float32,
        )

        with torch.no_grad():
            scaled_prediction = self.model(sequence_tensor)

        relative_prediction = self.y_scaler.inverse_transform(
            scaled_prediction.numpy().reshape(-1, 3)
        ).reshape(self.target_length, 3)

        return relative_prediction @ rotation + current_position


# Descriptive name used by the comparison code.  TargetPredictor remains
# available so the original single-method scripts keep working.
AccelerationTargetPredictor = TargetPredictor


__all__ = [
    "AccelerationTargetPredictor",
    "PositionTargetPredictor",
    "TargetPredictor",
    "LSTMModel",
    "PositionLSTMModel",
    "local_basis",
]
