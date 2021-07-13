import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

# class 형태의 모델은 nn.Module을 상속받음.
class BinaryClassifier(nn.Module):
    # 생성자 정의, 객체가 갖는 속성값을 초기화하는 역할.
    def __init__(self):
        # super 함수를 부르면 nn.Module 클래스의 속성들을 이용해 초기화 됨.
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    # model이 학습 데이터를 입력받아 forward 연산을 진행시키는 함수.
    # (forward : 입력 x로부터 예측된 y값을 얻는 것.)
    def forward(self, x):
        return self.sigmoid(self.linear(x))

# 생성자 호출.
model = BinaryClassifier()

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