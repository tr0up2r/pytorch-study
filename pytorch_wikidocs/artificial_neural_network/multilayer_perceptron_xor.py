import torch
import torch.nn as nn

# GPU 연산이 가능하다면 GPU 연산을 하도록.
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# for reproducibility
# 랜덤 시드 고정.
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

# XOR 문제를 풀기 위한 입력, 출력 정의.
X = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]]).to(device)
Y = torch.FloatTensor([[0], [1], [1], [0]]).to(device)

# 다층 퍼셉트론 설계.
# Hidden layer를 3개 가지는 인공 신경망.
model = nn.Sequential(
    nn.Linear(2, 10, bias=True), # input_layer = 2, hidden_layer1 = 10
    nn.Sigmoid(),
    nn.Linear(10, 10, bias=True), # hidden_layer1 = 10, hidden_layer2 = 10
    nn.Sigmoid(),
    nn.Linear(10, 10, bias=True), # hidden_layer2 = 10, hidden_layer3 = 10
    nn.Sigmoid(),
    nn.Linear(10, 1, bias=True), # hidden_layer3 = 10, output_layer = 1
    nn.Sigmoid()
).to(device)

# Cost function과 Optimizer 선언.
# nn.BCELoss() - binary classification에서 사용하는 크로스 엔트로피 함수.
criterion = torch.nn.BCELoss().to(device)

# modified learning rate from 0.1 to 1.
optimizer = torch.optim.SGD(model.parameters(), lr=1)

# 총 10,001번의 epoch을 수행.
# 각 epoch 마다 backpropagation이 수행된다.
for epoch in range(10001):
    optimizer.zero_grad()
    # forward 연산.
    hypothesis = model(X)

    # cost function.
    cost = criterion(hypothesis, Y)
    cost.backward()
    optimizer.step()

    # 100의 배수에 해당되는 epoch마다 cost를 출력.
    if epoch % 100 == 0:
        print(epoch, cost.item())

# 모델이 XOR 문제를 풀 수 있는지 테스트.
with torch.no_grad():
    hypothesis = model(X)
    predicted = (hypothesis > 0.5).float()
    accuracy = (predicted == Y).float().mean()
    print('모델의 출력값(Hypothesis): ', hypothesis.detach().cpu().numpy())
    print('모델의 예측값(Predicted): ', predicted.detach().cpu().numpy())
    print('실제값(Y): ', Y.cpu().numpy())
    print('정확도(Accuracy): ', accuracy.item())