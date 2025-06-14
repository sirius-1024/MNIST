import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision
import warnings
from thop import profile

warnings.filterwarnings("ignore")

# 超参数
EPOCH = 10
BATCH_SIZE = 50
LR = 0.001

prune_ratios = {
    'conv1': 0.5,
    'conv2': 0,
}

# === 加载数据 ===
train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
test_x = torch.unsqueeze(test_data.data, dim=1).float()[:2000] / 255.
test_y = test_data.targets[:2000]


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, 1, 2)
        self.conv2 = nn.Conv2d(16, 32, 5, 1, 2)
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        self.feat1 = x.clone()
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        self.feat2 = x.clone()
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        return self.out(x)


@torch.no_grad()
def evaluate(model):
    model.eval()
    output = model(test_x)
    pred_y = torch.max(output, 1)[1]
    return (pred_y == test_y).float().mean().item()


def compute_fisher_block(model, data_loader):
    model.eval()
    fisher_blocks = {
        'conv1': torch.zeros((16, 16)),
        'conv2': torch.zeros((32, 32)),
    }

    for x, _ in data_loader:
        model.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out, torch.randint(0, 10, (out.size(0),)))
        loss.backward()
        g1 = model.conv1.weight.grad.view(16, -1)
        g2 = model.conv2.weight.grad.view(32, -1)
        fisher_blocks['conv1'] += g1 @ g1.T / len(data_loader)
        fisher_blocks['conv2'] += g2 @ g2.T / len(data_loader)
    return fisher_blocks


def greedy_rearrange(fisher_block, orig_mask, num_keep):
    """
    贪婪调整 mask，保持保留通道数量不变，尽量降低残留 Fisher 信息。
    """
    mask = orig_mask.clone()
    assert mask.sum().item() == num_keep, f"初始保留通道数 {mask.sum().item()} != 目标 {num_keep}"

    best_score = fisher_block[~mask][:, ~mask].sum().item()

    for _ in range(5):
        improved = False
        pruned = torch.where(~mask)[0]  # 剪掉的通道
        kept = torch.where(mask)[0]     # 保留的通道

        for i in pruned:
            for j in kept:
                new_mask = mask.clone()
                new_mask[i] = True
                new_mask[j] = False
                score = fisher_block[~new_mask][:, ~new_mask].sum().item()
                if score < best_score:
                    mask = new_mask
                    best_score = score
                    improved = True
                    break  # 一旦有改进立刻更新，继续下一轮
            if improved:
                break

        if not improved:
            break

    assert mask.sum().item() == num_keep, f"最终保留通道数 {mask.sum().item()} != 目标 {num_keep}"
    return mask


def get_prune_masks(fisher_blocks, prune_ratios):
    masks = {}
    for name, fb in fisher_blocks.items():
        C = fb.size(0)
        ratio = prune_ratios.get(name, 0)
        num_prune = int(round(ratio * C))
        num_keep = C - num_prune

        print(f"[{name}] 总通道数: {C}, 目标保留: {num_keep}, 目标剪掉: {num_prune}")

        if num_prune == 0:
            masks[name] = torch.ones(C, dtype=torch.bool)
            continue

        importance = fb.diag()
        sorted_idx = torch.argsort(importance)   # 升序，最小的在前
        prune_idx = sorted_idx[:num_prune]       # 取最不重要的通道索引

        mask = torch.ones(C, dtype=torch.bool)
        mask[prune_idx] = False

        print(f"[{name}] mask初始保留通道数: {mask.sum().item()}, 保留通道索引: {torch.where(mask)[0].tolist()}")

        assert mask.sum().item() == num_keep, f"[{name}] 初始 mask 保留数量错误: {mask.sum().item()} != {num_keep}"

        masks[name] = greedy_rearrange(fb, mask, num_keep)

    return masks




class PrunedCNN(nn.Module):
    def __init__(self, original_model, masks):
        super().__init__()
        self.mask1 = masks['conv1']
        self.mask2 = masks['conv2']
        c1_out = self.mask1.sum().item()
        c2_out = self.mask2.sum().item()

        self.conv1 = nn.Conv2d(1, c1_out, 5, 1, 2)
        self.conv2 = nn.Conv2d(c1_out, c2_out, 5, 1, 2)
        self.out = nn.Linear(c2_out * 7 * 7, 10)

        with torch.no_grad():
            self.conv1.weight.data = original_model.conv1.weight[self.mask1].clone()
            self.conv1.bias.data = original_model.conv1.bias[self.mask1].clone()
            self.conv2.weight.data = original_model.conv2.weight[self.mask2][:, self.mask1].clone()
            self.conv2.bias.data = original_model.conv2.bias[self.mask2].clone()

            if original_model.out.weight.shape[1] == self.out.weight.shape[1]:
                self.out.weight.data = original_model.out.weight.clone()
                self.out.bias.data = original_model.out.bias.clone()
            else:
                nn.init.kaiming_normal_(self.out.weight, mode='fan_out')
                nn.init.zeros_(self.out.bias)

        self.scale1 = torch.ones(c1_out)
        self.scale2 = torch.ones(c2_out)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        self.feat1 = x.clone()
        x = x * self.scale1.reshape(1, -1, 1, 1)
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        self.feat2 = x.clone()
        x = x * self.scale2.reshape(1, -1, 1, 1)
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        return self.out(x)


def calculate_flops(model):
    dummy_input = torch.randn(1, 1, 28, 28)
    flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
    return flops


# === 实验开始 ===
print("\n===  剪枝实验设置 ===")
for k, v in prune_ratios.items():
    print(f"层 {k}: 剪枝比例 = {v:.1%}")

# 原始模型训练
cnn = CNN()
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

start = time.time()
print("\n===  开始训练原始模型 ===")
for epoch in range(EPOCH):
    for b_x, b_y in train_loader:
        optimizer.zero_grad()
        output = cnn(b_x)
        loss = loss_func(output, b_y)
        loss.backward()
        optimizer.step()
t_train = time.time() - start

# 评估原始模型
start = time.time()
original_accuracy = evaluate(cnn)
t_eval_orig = time.time() - start

# 计算 Fisher 信息
start = time.time()
fisher_blocks = compute_fisher_block(cnn, train_loader)
t_fisher = time.time() - start

# 生成剪枝 mask
start = time.time()
masks = get_prune_masks(fisher_blocks, prune_ratios)
t_mask = time.time() - start

# 构建剪枝模型并评估
start = time.time()
pruned_cnn = PrunedCNN(cnn, masks)
t_build_pruned = time.time() - start

start = time.time()
pruned_accuracy = evaluate(pruned_cnn)
t_eval_pruned = time.time() - start

# FLOPs / 参数统计
start = time.time()
original_flops = calculate_flops(cnn)
pruned_flops = calculate_flops(pruned_cnn)
original_params = sum(p.numel() for p in cnn.parameters())
pruned_params = sum(p.numel() for p in pruned_cnn.parameters())
t_flops = time.time() - start

# === 输出日志 ===
print("\n===  通道保留对比 ===")
print(f"{'层名':<10} | {'原始通道数':<10} | {'保留通道数':<10} | {'保留比例':<10}")
print("-" * 45)
print(f"{'conv1':<10} | {16:<14} | {masks['conv1'].sum().item():<14} | {masks['conv1'].sum().item()/16:<.2%}")
print(f"{'conv2':<10} | {32:<14} | {masks['conv2'].sum().item():<14} | {masks['conv2'].sum().item()/32:<.2%}")

print("\n===  准确率对比 ===")
print(f"{'模型':<10} | {'准确率':<10}")
print("-" * 25)
print(f"{'原始':<10} | {original_accuracy:<10.4f}")
print(f"{'剪枝后':<10} | {pruned_accuracy:<10.4f}")

print("\n===  计算开销对比 ===")
print(f"{'指标':<10} | {'原始':<15} | {'剪枝后':<15} | {'压缩率':<10}")
print("-" * 55)
print(f"{'FLOPs':<10} | {original_flops:<15,} | {pruned_flops:<15,} | {(1 - pruned_flops/original_flops):.2%}")
print(f"{'参数量':<10} | {original_params:<15,} | {pruned_params:<15,} | {(1 - pruned_params/original_params):.2%}")

print("\n===  时间消耗统计（单位: 秒） ===")
print(f"{'阶段':<20} | {'耗时'}")
print("-" * 35)
print(f"{'训练原始模型':<20} | {t_train:.2f}")
print(f"{'评估原始模型':<20} | {t_eval_orig:.2f}")
print(f"{'计算 Fisher 信息':<20} | {t_fisher:.2f}")
print(f"{'生成剪枝 mask':<20} | {t_mask:.2f}")
print(f"{'构建剪枝模型':<20} | {t_build_pruned:.2f}")
print(f"{'评估剪枝模型':<20} | {t_eval_pruned:.2f}")
print(f"{'计算 FLOPs/参数':<20} | {t_flops:.2f}")
