import argparse
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from model import PositionLSTMModel, local_basis
from sim import Sim
from vehicle import Vehicle

dt = 0.05
MODEL_DIR = Path(__file__).resolve().parents[1] / "model" / "3d" / "position"


def generate_dataset(number_of_simulations, duration):
    data_set = []

    for seed in tqdm(
        range(number_of_simulations),
        desc="3D 데이터 생성중",
    ):
        target = Vehicle(
            x=-1500.0,
            y=500.0,
            z=300.0,
            v=150.0,
        )

        sim = Sim(
            target=target,
            dt=dt,
            T_max=duration,
            seed=seed,
            maneuver_profile="standard",
        )

        data_set.append(sim.dataset_sim())

    return np.asarray(data_set, dtype=np.float32)


def create_sequence(
    data,
    sequence_length,
    target_length,
):
    sequences = []
    targets = []

    for episode in tqdm(
        data,
        desc="3D sequence 처리중",
    ):
        sample_count = len(episode) - sequence_length - target_length + 1

        for start in range(sample_count):
            history = episode[start : start + sequence_length]
            future = episode[
                start + sequence_length : start + sequence_length + target_length
            ]

            current_position = history[-1, 0:3]
            current_velocity = history[-1, 3:6]
            basis = local_basis(current_velocity)

            relative_position = (history[:, 0:3] - current_position) @ basis.T

            relative_velocity = history[:, 3:6] @ basis.T

            sequence = np.hstack(
                (
                    relative_position,
                    relative_velocity,
                )
            )

            future_position = (future[:, 0:3] - current_position) @ basis.T

            sequences.append(sequence)
            targets.append(future_position)

    return (
        np.asarray(sequences, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
    )


def create_loader(
    sequences,
    targets,
    batch_size,
    shuffle,
):
    dataset = TensorDataset(
        torch.tensor(
            sequences,
            dtype=torch.float32,
        ),
        torch.tensor(
            targets,
            dtype=torch.float32,
        ),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = generate_dataset(
        args.number_of_simulations,
        args.duration,
    )

    X, Y = create_sequence(
        data,
        args.sequence_length,
        args.target_length,
    )

    train_end = int(len(X) * 0.8)
    valid_end = int(len(X) * 0.9)

    x_train = X[:train_end]
    x_valid = X[train_end:valid_end]
    x_test = X[valid_end:]

    y_train = Y[:train_end]
    y_valid = Y[train_end:valid_end]
    y_test = Y[valid_end:]

    x_scaler = MinMaxScaler(feature_range=(-1, 1))
    y_scaler = MinMaxScaler(feature_range=(-1, 1))

    x_train_scaled = x_scaler.fit_transform(x_train.reshape(-1, 6)).reshape(
        x_train.shape
    )

    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 3)).reshape(
        y_train.shape
    )

    x_valid_scaled = x_scaler.transform(x_valid.reshape(-1, 6)).reshape(x_valid.shape)

    y_valid_scaled = y_scaler.transform(y_valid.reshape(-1, 3)).reshape(y_valid.shape)

    x_test_scaled = x_scaler.transform(x_test.reshape(-1, 6)).reshape(x_test.shape)

    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 3)).reshape(y_test.shape)

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        x_scaler,
        MODEL_DIR / "x_scaler.pkl",
    )
    joblib.dump(
        y_scaler,
        MODEL_DIR / "y_scaler.pkl",
    )

    train_loader = create_loader(
        x_train_scaled,
        y_train_scaled,
        args.batch_size,
        True,
    )
    valid_loader = create_loader(
        x_valid_scaled,
        y_valid_scaled,
        args.batch_size,
        False,
    )
    test_loader = create_loader(
        x_test_scaled,
        y_test_scaled,
        args.batch_size,
        False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PositionLSTMModel(
        input_size=6,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        target_length=args.target_length,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    best_validation_loss = float("inf")
    model_path = MODEL_DIR / "lstm_model.pth"

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
        )

        for sequences, targets in progress:
            sequences = sequences.to(device)
            targets = targets.to(device)

            predictions = model(sequences)
            loss = criterion(predictions, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.7f}")

        model.eval()
        validation_loss = 0.0

        with torch.no_grad():
            for sequences, targets in valid_loader:
                sequences = sequences.to(device)
                targets = targets.to(device)
                validation_loss += criterion(
                    model(sequences),
                    targets,
                ).item()

        validation_loss /= len(valid_loader)

        print(
            f"Epoch {epoch + 1:02d}, "
            f"Train: {train_loss / len(train_loader):.7f}, "
            f"Validation: {validation_loss:.7f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(model.state_dict(), model_path)

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device,
        )
    )
    model.eval()

    test_loss = 0.0

    with torch.no_grad():
        for sequences, targets in test_loader:
            sequences = sequences.to(device)
            targets = targets.to(device)
            test_loss += criterion(
                model(sequences),
                targets,
            ).item()

    print(f"Test loss: {test_loss / len(test_loader):.7f}")
    print(f"Model saved: {model_path}")


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--number-of-simulations",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=40,
    )
    parser.add_argument(
        "--target-length",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=96,
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0001,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    return parser.parse_args()


if __name__ == "__main__":
    train(arguments())
