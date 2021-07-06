import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# TorchVision에서 Fashion-MNIST 데이터 셋을 불러와서 사용하기
# 60,000개의 학습 예제, 10,000개의 테스트 예제
training_data = datasets.FashionMNIST(
    # 학습/테스트 데이터가 저장되는 경로
    root="data",

    # 학습용 또는 테스트용 데이터셋 여부를 지정
    train=True,

    # root에 데이터가 없는 경우, 인터넷에서 download
    download=True,

    # feature와 label transform을 지정
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)


# Dataset의 반복 + 시각화
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

# matplotlib을 이용한 시각화
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3

for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()