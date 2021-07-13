import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

# nn.Sequential - nn.Module 층을 차례로 쌓을 수 있도록 함.
model = nn.Sequential(
    # input_dim = 2, output_dim = 1
    nn.Linear(2, 1),
    # 출력은 시그모이드 함수를 거친다.
    nn.Sigmoid()
)

# W와 b는 현재 임의의 값을 가지므로, 예측에 의미가 없음.
print(model(x_train), '\n')

# gradient descent를 이용해 훈련.
# optimizer 설정
optimizer = optim.SGD(model.parameters(), lr=1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # H(x) 계산.
    hypothesis = model(x_train)

    # cost 계산.
    cost = F.binary_cross_entropy(hypothesis, y_train)

    # cost로 H(x) 개선.
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 20번마다 로그 출력.
    if epoch % 10 == 0:
        # 예측값이 0.5를 넘으면 True로 간주.
        prediction = hypothesis >= torch.FloatTensor([0.5])
        # 실제값과 일치하는 경우만 True로 간주.
        correct_prediction = prediction.float() == y_train
        # 정확도를 계산.
        accuracy = correct_prediction.sum().item() / len(correct_prediction)
        # 각 epoch마다 정확도를 출력.
        print('Epoch {:4d}/{} Cost: {:.6f} Accuracy {:2.2f}%'.format(
            epoch, nb_epochs, cost.item(), accuracy * 100,
                                           ))

# 중간부터 Accuracy가 100%로 나오기 시작함.

# 훈련 후의 W와 b의 값 출력.
# nn.Module을 사용하지 않고 얻었던 W와 b와 거의 일치함.
print(f'\n{list(model.parameters())}')
