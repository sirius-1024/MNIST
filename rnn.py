import torch
from torch import nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

EPOCH = 10
BATCH_SIZE = 50
TIME_STEP = 28
INPUT_SIZE = 28
LR = 0.01

train_data = datasets.MNIST(
    root='./mnist/',
    train=True,
    transform=transforms.ToTensor(),
    download=True
)

train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = datasets.MNIST(root='./mnist/', train=False, transform=transforms.ToTensor())
test_x = test_data.data.type(torch.FloatTensor)[:2000] / 255.
test_y = test_data.targets[:2000]

class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(input_size=INPUT_SIZE, hidden_size=64, num_layers=1, batch_first=True)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        r_out, _ = self.rnn(x)
        out = self.fc(r_out[:, -1, :])
        return out

model = RNN()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        x = x.view(-1, 28, 28)
        out = model(x)
        loss = loss_func(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_out = model(test_x.view(-1, 28, 28))
            pred = torch.max(test_out, 1)[1]
            acc = (pred == test_y).float().mean()
            print(f'Epoch: {epoch} | train loss: {loss.item():.4f} | test accuracy: {acc:.2f}')

test_out = model(test_x[:10].view(-1, 28, 28))
pred = torch.max(test_out, 1)[1]
print('Prediction:', pred.tolist())
print('Actual:', test_y[:10].tolist())
