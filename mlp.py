import os
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision

EPOCH = 10
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = False

if not (os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    DOWNLOAD_MNIST = True

train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST,
)

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:2000] / 255.
test_y = test_data.targets[:2000]

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(28 * 28, 28 * 28),
            nn.ReLU(),
            nn.Linear(28 * 28, 10)
        )

    def forward(self, x):
        x = self.mlp(x)
        return x, x

mlp = MLP()
optimizer = torch.optim.Adam(mlp.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):
        b_x = b_x.view(-1, 28 * 28)

        output, _ = mlp(b_x)
        loss = loss_func(output, b_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_output, _ = mlp(test_x.view(-1, 28 * 28))
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y.data.numpy()).sum()) / test_y.size(0)
            print('Epoch:', epoch, '| Loss: %.4f' % loss.item(), '| Test Acc: %.2f' % accuracy)

test_output, _ = mlp(test_x[:10].view(-1, 28 * 28))
pred_y = torch.max(test_output, 1)[1].data.numpy()
print('Prediction:', pred_y)
print('Actual:    ', test_y[:10].numpy())
