import torch
import torch.optim as optim

# 랜덤 시드 고정.
torch.manual_seed(1)

# x의 개수가 3개이므로, 훈련 데이터 선언시 x 3개 선언.
x1_train = torch.FloatTensor([[73], [93], [89], [96], [73]])
x2_train = torch.FloatTensor([[80], [88], [91], [98], [66]])
x3_train = torch.FloatTensor([[75], [93], [90], [100], [70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

# weight와 bias도 3개씩 선언.
w1 = torch.zeros(1, requires_grad=True)
w2 = torch.zeros(1, requires_grad=True)
w3 = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# optimizer 설정.
optimizer = optim.SGD([w1, w2, w3, b], lr=1e-5)

# gradient desecent 1,000회 반복.
nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # H(x) 계산.
    hypothesis = x1_train * w1 + x2_train * w2 + x3_train * w3 + b

    # cost fucntion 계산.
    cost = torch.mean((hypothesis - y_train) ** 2)

    # cost로 H(x) 개선.
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력.
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} w1: {:.3f} w2: {:.3f} w3: {:.3f} b: {:.3f} Cost: {:.6f}'.format(
            epoch, nb_epochs, w1.item(), w2.item(), w3.item(), b.item(), cost.item()
        ))