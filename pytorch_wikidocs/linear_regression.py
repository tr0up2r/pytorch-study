import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

# x_train과 x_train의 크기(shape) 출력.
print(x_train)
print(x_train.shape, '\n')

# y_train과 y_train의 크기(shape) 출력.
print(y_train)
print(y_train.shape, '\n')

# 가중치 W를 0으로 초기화.
# requires_grad=True -> 학습을 통해 값이 변경되는 변수임을 명시.
# PyTorch에서 제공하는 Autograd 기능을 수행하고 있는 것. (1)
W = torch.zeros(1, requires_grad=True)
# 가중치 W를 출력.
print('W = ', W)
# bias를 0으로 초기화 후 출력.
b = torch.zeros(1, requires_grad=True)
print('b =', b, '\n')

# hypothesis 세우기.
hypothesis = x_train * W + b
print('hypothesis =', hypothesis)

# cost function 선언하기.
# 평균 제곱 오차를 선언.
cost = torch.mean((y_train - hypothesis) ** 2)
print('cost =', cost)


# gradient descent 구현하기. (with SGD)
# lr = learning rate
optimizer = optim.SGD([W, b], lr=0.01)

# 원하는만큼 gradient descent를 반복하기 위한 epoch 설정.
nb_epochs = 2000
for epoch in range(nb_epochs + 1):
    # H(x) 계산.
    hypothesis = x_train * W + b

    # cost 계산.
    cost = torch.mean((hypothesis - y_train) ** 2)

    # cost로 H(x) 개선.
    # gradient를 0으로 초기화.
    # 이 부분을 하지 않으면 계속해서 미분값이 누적이 되므로, epoch마다 0으로 초기화.
    optimizer.zero_grad()

    # cost function을 미분하여 gradient 계산.
    # PyTorch에서 제공하는 Autograd 기능을 수행하고 있는 것. (2)
    cost.backward()
    # W와 b의 값을 업데이트.
    optimizer.step()

    # 100번마다 log 출력.
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
            epoch, nb_epochs, W.item(), b.item(), cost.item()
        ))