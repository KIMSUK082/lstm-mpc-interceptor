import torch.nn as nn
import torch
import joblib
import numpy as np


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size  ## lstm의 ht와 cellstate의 값의 개수
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
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
        ## mlp는 데이터를 펼쳐서 줘야하고 출력값도 펼쳐서 나옴
        ## batch size가 꼭 64가 아닐수 있음 x.size는 그당시에 batchsize
        out = out.reshape(x.size(0), 20, 2)

        return out


class TargetPredictor:
    def __init__(
        self,
        model_path="lstm_model.pth",
        x_scaler_path="x_scaler.pkl",
        y_scaler_path="y_scaler.pkl",
        input_size=4,
        hidden_size=64,
        num_layers=2,
        output_size=40,
    ):
        self.x_scaler = joblib.load(x_scaler_path)
        self.y_scaler = joblib.load(y_scaler_path)

        self.model = LSTMModel(input_size, hidden_size, num_layers, output_size)
        self.model.load_state_dict(
            torch.load(model_path, map_location=torch.device("cpu"))
        )
        self.model.eval()

    def to_absolute(self, rel_pred, current_pos, current_vel):
        heading = np.arctan2(current_vel[1], current_vel[0])
        R = np.array(
            [
                [np.cos(heading), np.sin(heading)],
                [-np.sin(heading), np.cos(heading)],
            ]
        )
        ## 상대좌표를 절대좌표로 변환
        return rel_pred @ R + current_pos

    def predict(self, history):
        hist = np.array(history)
        current_pos = hist[-1, 0:2]
        current_vel = hist[-1, 2:]
        heading = np.arctan2(current_vel[1], current_vel[0])
        R = np.array(
            [
                [np.cos(heading), np.sin(heading)],
                [-np.sin(heading), np.cos(heading)],
            ]
        )

        ## LSTM.py의 create_seqeunce와 동일한 상대좌표 변환
        delta_pos = hist[:, 0:2] - current_pos
        rel_pos = delta_pos @ R.T
        rel_vel = hist[:, 2:] @ R.T
        seq = np.hstack((rel_pos, rel_vel))

        seq_scaled = self.x_scaler.transform(seq).reshape(1, 40, 4)
        seq_tensor = torch.tensor(seq_scaled, dtype=torch.float32)

        with torch.no_grad():
            pred_scaled = self.model(seq_tensor)  # (1, 20, 2)

        pred_rel = self.y_scaler.inverse_transform(
            pred_scaled.numpy().reshape(-1, 2)
        ).reshape(20, 2)

        return self.to_absolute(pred_rel, current_pos, current_vel)
