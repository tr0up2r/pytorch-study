import os
import torch

# torch.nn -> 신경망을 구성할 때 필요한 모든 구성 요소를 제공
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 가능하면 GPU와 같은 하드웨어 가속기로 모델을 학습하려 함
# 불가능하면 CPU를 계속 이용
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))


# 신경망 모델을 nn.Module의 하위 클래스로 정의
class NeuralNetwork(nn.Module):
    # 신경망 계층들을 초기화
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        # 2차원 데이터를 1차원 데이터로 바꾸는 역할의 레이어
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    # 입력 데이터에 대한 연산들을 구현
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# NeuralNetwork의 instance를 생성
# device로 이동한 뒤 structure를 출력
model = NeuralNetwork().to(device)
print(model, '\n')



# model을 사용하기 위해 입력 데이터를 전달
# 난수로 이루어진 Tensor를 생성.
X = torch.rand(1, 28, 28, device=device)

# X라는 입력을 parameter로 넘겨주면,
# class에 대한 raw 예측값이 있는 10차원 tensor가 반환됨
# why 10차원...?
logits = model(X)

# raw 예측값 출력해보기
print(logits)

# raw 예측값을 nn.Softmax 모듈의 instance를 통과시켜
# 0~1 사이의 최종 예측 확률을 얻음
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)

# 최종 예측 label을 출력!
print(f"Predicted class: {y_pred}\n\n")



# FashionMNIST 모델의 계층 살펴보기
# 28x28 크기의 이미지 3개로 구성된 미니배치를 가져옴
input_image = torch.rand(3,28,28)
print(input_image.size(), '\n')


# nn.Flatten
# 28x28의 2D 이미지를 784 픽셀 값을 갖는 연속 배열로 반환
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size(), '\n')


# nn.Linear
# weight와 bias를 사용해 입력에 linear transformation을 적용
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size(), '\n')


# nn.ReLU
# 모델의 입력과 출력 사이의 관계를 mapping
# 신경망이 다양한 현상을 학습할 수 있도록 도움
print(f"Before ReLU: {hidden1}\n")
hidden1 = nn.ReLU()(hidden1)

# 얘도 약간 softmax처럼 값이 변환됨.
# 음수였던 값들은 무조건 0으로
print(f"After ReLU: {hidden1}\n\n")


# nn.Sequential
# sequential container를 이용해 신경망을 빠르게 만들 수 있음
# 아래와 같이 정해진 순서로 데이터들이 전달됨
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)


# nn.Softmax
# 신경망의 마지막 선형 계층
# logits는 모델의 각 class에 대한 예측 확률을 나타내도록
# [0, 1] 범위로 비례하여 scale됨
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)
print(pred_probab, '\n')