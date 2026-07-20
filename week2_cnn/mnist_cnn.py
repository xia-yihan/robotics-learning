import os

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# =========================
# 1. 基本设置
# =========================

torch.manual_seed(42)

BATCH_SIZE = 128
EPOCHS = 3
LEARNING_RATE = 0.05

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("output", exist_ok=True)

print("使用设备：", device)


# =========================
# 2. 准备数据
# =========================

transform = transforms.ToTensor()

full_train_set = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)

test_set = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)

# 为了加快演示，只使用部分训练数据
train_set, val_set, _ = random_split(
    full_train_set,
    [20000, 5000, 35000],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("训练集数量：", len(train_set))
print("验证集数量：", len(val_set))
print("测试集数量：", len(test_set))


# =========================
# 3. 定义小型 CNN
# =========================

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = SmallCNN().to(device)

# 交叉熵用于多分类任务
loss_function = nn.CrossEntropyLoss()

# SGD 根据梯度更新模型参数
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=LEARNING_RATE,
    momentum=0.9
)


# =========================
# 4. 训练和验证函数
# =========================

def train_one_epoch():
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # 清空上一轮梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(images)

        # 计算损失
        loss = loss_function(outputs, labels)

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


def evaluate(data_loader):
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_function(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


# =========================
# 5. 正式训练
# =========================

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(EPOCHS):
    train_loss, train_accuracy = train_one_epoch()
    val_loss, val_accuracy = evaluate(val_loader)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"训练损失：{train_loss:.4f} | "
        f"训练准确率：{train_accuracy:.4f} | "
        f"验证损失：{val_loss:.4f} | "
        f"验证准确率：{val_accuracy:.4f}"
    )


# =========================
# 6. 保存模型
# =========================

model_path = "output/mnist_cnn.pth"

torch.save(model.state_dict(), model_path)

print("模型已保存：", model_path)


# =========================
# 7. 绘制训练曲线
# =========================

epochs = range(1, EPOCHS + 1)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, marker="o", label="Train loss")
plt.plot(epochs, val_losses, marker="o", label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, marker="o", label="Train accuracy")
plt.plot(epochs, val_accuracies, marker="o", label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("output/training_curves.png", dpi=150)
plt.close()

print("训练曲线已保存")


# =========================
# 8. 重新加载模型
# =========================

loaded_model = SmallCNN().to(device)

state_dict = torch.load(
    model_path,
    map_location=device,
    weights_only=True
)

loaded_model.load_state_dict(state_dict)
loaded_model.eval()

print("模型重新加载成功")


# =========================
# 9. 测试集与混淆矩阵
# =========================

confusion_matrix = torch.zeros(10, 10, dtype=torch.int64)
error_samples = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = loaded_model(images.to(device))
        predictions = outputs.argmax(dim=1).cpu()

        indexes = labels * 10 + predictions

        confusion_matrix += torch.bincount(
            indexes,
            minlength=100
        ).reshape(10, 10)

        wrong_mask = predictions != labels

        for image, true_label, predicted_label in zip(
            images[wrong_mask],
            labels[wrong_mask],
            predictions[wrong_mask]
        ):
            if len(error_samples) < 12:
                error_samples.append(
                    (image.squeeze(0), true_label.item(), predicted_label.item())
                )

correct = confusion_matrix.diag().sum().item()
total = confusion_matrix.sum().item()
test_accuracy = correct / total

print(f"测试集准确率：{test_accuracy:.4f}")


# 绘制混淆矩阵
plt.figure(figsize=(8, 7))
plt.imshow(confusion_matrix, cmap="Blues")
plt.colorbar()

plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion Matrix")

plt.xticks(range(10))
plt.yticks(range(10))

for row in range(10):
    for column in range(10):
        value = int(confusion_matrix[row, column])

        plt.text(
            column,
            row,
            value,
            ha="center",
            va="center",
            fontsize=7
        )

plt.tight_layout()
plt.savefig("output/confusion_matrix.png", dpi=150)
plt.close()

print("混淆矩阵已保存")


# =========================
# 10. 展示错误样本
# =========================

plt.figure(figsize=(10, 7))

for index, (image, true_label, predicted_label) in enumerate(error_samples):
    plt.subplot(3, 4, index + 1)
    plt.imshow(image, cmap="gray")
    plt.title(f"True: {true_label}, Pred: {predicted_label}")
    plt.axis("off")

plt.tight_layout()
plt.savefig("output/error_samples.png", dpi=150)
plt.close()

print("错误样本图已保存")


# =========================
# 11. 单张图片推理
# =========================

sample_image, sample_label = test_set[0]

with torch.no_grad():
    prediction = loaded_model(
        sample_image.unsqueeze(0).to(device)
    ).argmax(dim=1).item()

print("单张图片真实标签：", sample_label)
print("单张图片预测标签：", prediction)
print("全部任务完成")