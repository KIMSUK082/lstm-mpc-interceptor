import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sim import Sim
from vehicle import Vehicle

N_SIMS = 1000
seq_length = 40
target_length = 20
input_size = 4
output_size = 40
hidden_size = 64
batch_size = 64
num_layers = 2
epochs = 30
lr = 0.0001


## lstm을 절대좌표로 학습시키면 절대좌표 그자체를 외워버리기에 학습률이 떨어짐
## 표적의 움직임을 반영하여 학습시키려면 상대위치 상대속도를 학습시켜야함 표적 현재시점을 기준으로
## 표적의 과거 위치를 상대위치로 변환 이러면 물체가 움직임만을 학습할수있음
## 먼저 상대정규화후 -1~1로 minmax정규화 움직임은 음수도 포함하고 있어야함

data_set = []

for i in tqdm(range(N_SIMS), desc="데이터 생성중"):
    target = Vehicle(-1500, 500, v=150.0)
    sim = Sim(target, T_max=30, dt=0.05, seed=i)
    data = sim.dataset_sim()
    data_set.append(data)
data_set = np.array(data_set)
print(data_set.shape)


def create_seqeunce(data, seq_length, target_length):
    sequences = []
    targets = []
    for i in tqdm(range(N_SIMS), desc="데이터 처리중"):
        episode = data[i]
        for k in range(len(episode) - seq_length - target_length + 1):
            history = episode[k : k + seq_length]
            future = episode[k + seq_length : (k + seq_length + target_length)]
            current_state = history[-1]  ## -1은 마지막값
            current_pos = current_state[0:2]
            current_vel = current_state[2:]
            heading = np.arctan2(current_vel[1], current_vel[0])
            R = np.array(
                [
                    [np.cos(heading), np.sin(heading)],
                    [-np.sin(heading), np.cos(heading)],
                ]
            )

            ## sequence 구하기
            delta_pos = history[:, 0:2] - current_pos
            rel_pos = delta_pos @ R.T
            rel_vel = history[:, 2:] @ R.T
            sequences.append(np.hstack((rel_pos, rel_vel)))

            ## target 구하기
            delta_pos = future[:, 0:2] - current_pos
            rel_pos = delta_pos @ R.T
            targets.append(rel_pos)

    return np.array(sequences), np.array(targets)


X, Y = create_seqeunce(data_set, seq_length, target_length)

total_size = len(X)
train_end = int(total_size * 0.8)
valid_end = int(total_size * 0.9)

x_train, x_valid, x_test = X[:train_end], X[train_end:valid_end], X[valid_end:]
y_train, y_valid, y_test = Y[:train_end], Y[train_end:valid_end], Y[valid_end:]


## 스케일러를 다르게 두어 sequence와 target을 각각 정규화
x_scaler = MinMaxScaler(feature_range=(-1, 1))
y_scaler = MinMaxScaler(feature_range=(-1, 1))

## minmaxscaler는 2차원 데이터밖에 못받음 그래서 reshape로 정규화후 scale
## train data만 fit
x_train_scaled = x_scaler.fit_transform(x_train.reshape(-1, 4)).reshape(x_train.shape)
y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 2)).reshape(y_train.shape)

x_valid_scaled = x_scaler.transform(x_valid.reshape(-1, 4)).reshape(x_valid.shape)
y_valid_scaled = y_scaler.transform(y_valid.reshape(-1, 2)).reshape(y_valid.shape)

x_test_scaled = x_scaler.transform(x_test.reshape(-1, 4)).reshape(x_test.shape)
y_test_scaled = y_scaler.transform(y_test.reshape(-1, 2)).reshape(y_test.shape)


## tensor화
x_train = torch.tensor(x_train_scaled, dtype=torch.float32)
y_train = torch.tensor(y_train_scaled, dtype=torch.float32)
x_test = torch.tensor(x_test_scaled, dtype=torch.float32)
y_test = torch.tensor(y_test_scaled, dtype=torch.float32)
x_valid = torch.tensor(x_valid_scaled, dtype=torch.float32)
y_valid = torch.tensor(y_valid_scaled, dtype=torch.float32)

train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)
valid_dataset = TensorDataset(x_valid, y_valid)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
)

val_loader = DataLoader(
    valid_dataset,
    batch_size=batch_size,
    shuffle=False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
)

data_iter = iter(test_loader)
sequences, targets = next(data_iter)

# 2. 형태(Shape)나 값을 확인해봅니다.
print("Sequence Shape:", sequences.shape)
print("Target Shape:", targets.shape)


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


model = LSTMModel(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

model.train()
for epoch in range(epochs):
    train_loss = 0.0
    ## tqdm은 진행바를 띄우게 하는 라이브러리 desc는 옆에 문자열을 출력하게함
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
    for sequences, targets in pbar:
        output = model(sequences)
        loss = criterion(output, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        ##tqdm set_postfix 진행바 뒤에 loss률을 띄움
        pbar.set_postfix(loss=f"{loss.item():.7f}")
    print(
        f"Epoch {str(epoch+1).zfill(2)}/{epochs}, Loss: {train_loss/len(train_loader):.7f}"
    )

    model.eval()

    with torch.no_grad():
        val_loss = 0.0
        for sequences, targets in val_loader:
            output = model(sequences)
            loss = criterion(output, targets)
            val_loss += loss.item()

        val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {val_loss:.7f}")

    model.train()
