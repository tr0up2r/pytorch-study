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

# model 선언 및 초기화.
# 4개의 feature를 가지고 3개의 class로 분류.
# input_dim = 4, output_dim = 3.
model = nn.Linear(4, 3)

# optimizer 설정
optimizer = optim.SGD(model.parameters(), lr=0.1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # H(x) 계산.
    prediction = model(x_train)

    # cost 계산.
    # F.cross_entropy()를 사용하므로, softmax 함수를 가설에 따로 정의 안 함.
    cost = F.cross_entropy(prediction, y_train)

    # cost로 H(x) 개선.
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 20번마다 로그 출력.
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))