# 매개변수를 최적화하여 model을 학습, 검증, 테스트 하기
# 반복적인 과정(epoch)을 거치며
# 출력을 추측하고, 추측과 정답 사이의 오류(loss)를 계산하고
# 매개변수에 대한 loss의 도함수를 수집한 뒤
# 경사하강법을 사용하여 이 parameter들을 optimize함

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

# Pre-requisite 코드들
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork()

# Hyperparameter
# 모델 최적화 과정을 제어할 수 있는 조절 가능한 매개변수

# 학습율 - 각 배치/epoch에서 모델의 매개변수를 조절하는 비율
# 값이 작음 - 학습 속도가 느려짐
# 값이 큼 - 학습 중 예측할 수 없는 동작이 발생할 수 있음
learning_rate = 1e-3

# 배치 크기 - 매개변수가 갱신되기 전 신경망을 통해 전파된 데이터 샘플 수
batch_size = 64

# epoch 수 - 데이터 셋을 반복해서 학습하는 횟수
epochs = 5


# Optimization Loop
# 하나의 Epoch은 train loop + validation/test loop 으로 구성됨
# train loop - 학습용 데이터셋을 반복(iterate)하고 최적의 매개변수로 수렴
# validation/test loop - 모델 성능이 개선되고 있는지를 확인하기 위해 테스트 데이터셋을 반복(iterate)

# loss function
# 획득한 예측 결과와 실제 값 사이의 틀린 정도를 측정
# 학습 중, 이 값을 최소화하려고 함
# 손실 함수 초기화
loss_fn = nn.MSELoss() # 평균 제곱 오차(Mean Square Error), regression
loss_fn = nn.NLLLoss() # 음의 로그 우도(Negative Log Likelihood), classification
loss_fn = nn.CrossEntropyLoss() # 크로스 엔트로피(nn.LogSoftmax + nn.NLLLoss)

# Optimizer
# 다양한 옵티마이저 존재
# 여기서는 확률적 경사하강법(Stochastic Gradient Descent)를 채택, 정의
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# 최적화 코드를 반복하여 수행
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # 예측(prediction)과 손실(loss) 계산
        pred = model(X)
        loss = loss_fn(pred, y)

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# test data로 model의 성능을 측정
def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 모델의 성능 향상을 알아보기 위해 epoch 수를 자유롭게 증가시켜 볼 수 있음.
epochs = 10

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")