import torch
import torch.nn.functional as F
import torch.optim as optim

# seed 값 설정.
torch.manual_seed(1)

x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

print('x_train shape :', x_train.shape)
print('y_train shape :', y_train.shape, '\n')

# x_train -> 6 * 2 크기의 행렬.
# y_train -> 6 * 1 크기의 벡터.
# XW가 성립되기 위해서는 W 벡터의 크기는 2 * 1이어야 함.
W = torch.zeros((2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 지수 함수 사용을 위한 torch.exp(x).
hypothesis = 1 / (1 + torch.exp(-(x_train.matmul(W) + b)))

# 위처럼 해도 되지만, PyTorch에서는 sigmoid를 바로 지원.
hypothesis = torch.sigmoid(x_train.matmul(W) + b)
print('hypothesis :', hypothesis, '\n')

# Logistic regression의 cost function을 이미 구현해서 제공 중.
losses = F.binary_cross_entropy(hypothesis, y_train)
print('losses :', losses, '\n')


# 모델 훈련 과정.

# optimizer 설정.
optimizer = optim.SGD([W, b], lr=1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # Cost 계산
    hypothesis = torch.sigmoid(x_train.matmul(W) + b)
    cost = -(y_train * torch.log(hypothesis) +
             (1 - y_train) * torch.log(1 - hypothesis)).mean()

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))

# 학습 완료 후, 제대로 예측하는지를 확인.
hypothesis = torch.sigmoid(x_train.matmul(W) + b)
print('sigmoid hypothesis :', hypothesis)

# 0과 1 사이의 값이 아닌, 0.5를 넘으면 True, 넘지 않으면 False로 출력해보기.
prediction = hypothesis >= torch.FloatTensor([0.5])
print('prediction :\n', prediction, '\norigin y :\n', y_train, '\n')

print(f'W : {W},\nb : {b}')