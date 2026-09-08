from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from sim import Sim
from vehicle import Vehicle


N_SIMS = 500
seq_length = 40
target_length = 8

input_size = 6
output_size = 24
hidden_size = 64
num_layers = 2

batch_size = 64
epochs = 30
lr = 0.0001
dt = 0.05

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_DIR / "model" / "3d" / "acceleration"
RESULT_DIR = PROJECT_DIR / "results_3d"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "lstm_model.pth"
X_SCALER_PATH = MODEL_DIR / "x_scaler.pkl"
Y_SCALER_PATH = MODEL_DIR / "y_scaler.pkl"


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


def create_data_set(n_sims, seed_offset=0):
    data_set = []
    initial_rng = np.random.default_rng(seed_offset)

    for index in tqdm(range(n_sims), desc="데이터 생성중"):
        target = Vehicle(
            x=-1500.0,
            y=500.0,
            z=1000.0,
            v=initial_rng.uniform(130.0, 170.0),
            flight_path_angle=np.radians(
                initial_rng.uniform(-10.0, 10.0)
            ),
            heading=initial_rng.uniform(-np.pi, np.pi),
        )

        sim = Sim(
            target,
            T_max=20.0,
            dt=dt,
            seed=seed_offset + index,
            maneuver_profile="dynamic",
        )

        data = sim.dataset_sim()
        data_set.append(data)

    return np.asarray(data_set, dtype=np.float32)


def create_sequence(data, seq_length, target_length):
    sequences = []
    targets = []

    for episode in tqdm(data, desc="데이터 처리중"):
        sample_count = (
            len(episode)
            - seq_length
            - target_length
            + 1
        )

        for index in range(sample_count):
            history = episode[index : index + seq_length]
            future = episode[
                index + seq_length :
                index + seq_length + target_length
            ]

            current_state = history[-1]
            current_position = current_state[0:3]
            current_velocity = current_state[3:6]

            rotation = local_basis(current_velocity)

            ## 현재 위치와 진행 방향을 기준으로 입력을 상대좌표화
            delta_position = history[:, 0:3] - current_position
            relative_position = delta_position @ rotation.T
            relative_velocity = history[:, 3:6] @ rotation.T

            sequences.append(
                np.hstack(
                    (
                        relative_position,
                        relative_velocity,
                    )
                )
            )

            ## 미래 속도의 차이로 실제 가속도를 계산
            future_velocity = future[:, 3:6]
            velocity_sequence = np.vstack(
                (
                    current_velocity,
                    future_velocity,
                )
            )

            global_acceleration = (
                np.diff(velocity_sequence, axis=0) / dt
            )

            start_velocities = velocity_sequence[:-1]
            local_acceleration = []

            ## 각 미래 시점의 진행 방향으로 가속도를 분해
            for acceleration, velocity in zip(
                global_acceleration,
                start_velocities,
            ):
                future_rotation = local_basis(velocity)
                local_acceleration.append(
                    acceleration @ future_rotation.T
                )

            targets.append(local_acceleration)

    return (
        np.asarray(sequences, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
    )


def make_loader(x_data, y_data, shuffle):
    x_tensor = torch.tensor(x_data, dtype=torch.float32)
    y_tensor = torch.tensor(y_data, dtype=torch.float32)

    dataset = TensorDataset(x_tensor, y_tensor)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )


class LSTMModel(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        output_size,
    ):
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

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
            target_length,
            3,
        )


def train_model(model, train_loader, valid_loader, device):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    train_losses = []
    valid_losses = []
    best_validation_loss = np.inf

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs}",
        )

        for sequences, targets in progress:
            sequences = sequences.to(device)
            targets = targets.to(device)

            output = model(sequences)
            loss = criterion(output, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            progress.set_postfix(
                loss=f"{loss.item():.7f}"
            )

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        validation_loss = 0.0

        with torch.no_grad():
            for sequences, targets in valid_loader:
                sequences = sequences.to(device)
                targets = targets.to(device)

                output = model(sequences)
                loss = criterion(output, targets)
                validation_loss += loss.item()

        validation_loss /= len(valid_loader)
        valid_losses.append(validation_loss)

        print(
            f"Epoch {epoch + 1:02d}/{epochs}, "
            f"Loss: {train_loss:.7f}, "
            f"Validation Loss: {validation_loss:.7f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(model.state_dict(), MODEL_PATH)

    return train_losses, valid_losses


def evaluate_model(model, data_loader, y_scaler, device):
    predictions = []
    targets = []

    model.eval()

    with torch.no_grad():
        for sequences, target in tqdm(
            data_loader,
            desc="학습",
        ):
            output = model(sequences.to(device))

            predictions.append(output.cpu().numpy())
            targets.append(target.numpy())

    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)

    predictions = y_scaler.inverse_transform(
        predictions.reshape(-1, 3)
    ).reshape(predictions.shape)

    targets = y_scaler.inverse_transform(
        targets.reshape(-1, 3)
    ).reshape(targets.shape)

    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    mae = np.mean(np.abs(predictions - targets))

    return rmse, mae, predictions, targets


def main():
    ## 같은 시뮬레이션이 train과 test에 동시에 들어가지 않도록 episode 단위로 분리
    data_set = create_data_set(N_SIMS)

    train_end = int(len(data_set) * 0.8)
    valid_end = int(len(data_set) * 0.9)

    train_data = data_set[:train_end]
    valid_data = data_set[train_end:valid_end]
    test_data = data_set[valid_end:]

    x_train, y_train = create_sequence(
        train_data,
        seq_length,
        target_length,
    )
    x_valid, y_valid = create_sequence(
        valid_data,
        seq_length,
        target_length,
    )
    x_test, y_test = create_sequence(
        test_data,
        seq_length,
        target_length,
    )

    x_scaler = MinMaxScaler(feature_range=(-1, 1))
    y_scaler = MaxAbsScaler()

    x_train_scaled = x_scaler.fit_transform(
        x_train.reshape(-1, input_size)
    ).reshape(x_train.shape)

    y_train_scaled = y_scaler.fit_transform(
        y_train.reshape(-1, 3)
    ).reshape(y_train.shape)

    x_valid_scaled = x_scaler.transform(
        x_valid.reshape(-1, input_size)
    ).reshape(x_valid.shape)

    y_valid_scaled = y_scaler.transform(
        y_valid.reshape(-1, 3)
    ).reshape(y_valid.shape)

    x_test_scaled = x_scaler.transform(
        x_test.reshape(-1, input_size)
    ).reshape(x_test.shape)

    y_test_scaled = y_scaler.transform(
        y_test.reshape(-1, 3)
    ).reshape(y_test.shape)

    joblib.dump(x_scaler, X_SCALER_PATH)
    joblib.dump(y_scaler, Y_SCALER_PATH)

    train_loader = make_loader(
        x_train_scaled,
        y_train_scaled,
        shuffle=True,
    )
    valid_loader = make_loader(
        x_valid_scaled,
        y_valid_scaled,
        shuffle=False,
    )
    test_loader = make_loader(
        x_test_scaled,
        y_test_scaled,
        shuffle=False,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"학습 장치: {device}")

    model = LSTMModel(
        input_size,
        hidden_size,
        num_layers,
        output_size,
    ).to(device)

    train_losses, valid_losses = train_model(
        model,
        train_loader,
        valid_loader,
        device,
    )

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device)
    )

    test_rmse, test_mae, _, _ = evaluate_model(
        model,
        test_loader,
        y_scaler,
        device,
    )

    print(f"Test acceleration RMSE: {test_rmse:.4f} m/s²")
    print(f"Test acceleration MAE: {test_mae:.4f} m/s²")
    print(f"학습 및 모델 저장 완료: {MODEL_PATH}")

    epoch_axis = np.arange(1, epochs + 1)

    plt.figure(figsize=(9, 5))
    plt.plot(epoch_axis, train_losses, label="Train loss")
    plt.plot(epoch_axis, valid_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Scaled acceleration MSE")
    plt.title("3D LSTM Train and Validation Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        RESULT_DIR / "train_validation_loss.png",
        dpi=200,
    )
    plt.show()


if __name__ == "__main__":
    main()
