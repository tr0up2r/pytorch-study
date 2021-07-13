import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

# x_train의 각 샘플은 4개의 feature를 가짐.
# 총 8개의 샘플이 존재.
x_train = [[1, 2, 1, 1],
           [2, 1, 3, 2],
           [3, 1, 3, 4],
           [4, 1, 5, 5],
           [1, 7, 5, 5],
           [1, 2, 5, 6],
           [1, 6, 6, 6],
           [1, 7, 7, 7]]

# 각 샘플에 대한 label.
# 0, 1, 2 값인 것으로 보아 총 3개의 클래스.
y_train = [2, 2, 2, 1, 1, 1, 0, 0]

# 모델 초기화
W = torch.zeros((4, 3), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# optimizer 설정
optimizer = optim.SGD([W, b], lr=0.1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # Cost 계산.
    z = x_train.matmul(W) + b

    # F.cross_entropy 그 자체로 softmax function을 포함하고 있음.
    # 따라서, hypothesis에서 softmax 함수 따로 사용할 필요 없음.
    cost = F.cross_entropy(z, y_train)

    # cost로 H(x) 개선.
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력.
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))