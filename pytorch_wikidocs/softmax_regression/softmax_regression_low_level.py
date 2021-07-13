import torch
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

x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)

print(x_train.shape)
print(y_train.shape, '\n')

# x_train의 크기 8 * 4, y_train의 크기 8 * 1
# y_train에 원-핫 인코딩한 결과는 8 * 3의 개수를 가져야 함.
y_one_hot = torch.zeros(8, 3)
y_one_hot.scatter_(1, y_train.unsqueeze(1), 1)
print(y_one_hot.shape, '\n')

# 모델 초기화
# W 행렬의 크기는 y_one_hot이 8 * 3이므로, 4 * 3이 됨.
W = torch.zeros((4, 3), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# optimizer 설정
optimizer = optim.SGD([W, b], lr=0.1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # 가설 설정.
    hypothesis = F.softmax(x_train.matmul(W) + b, dim=1)

    # cost function 설정.
    cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()

    # cost로 H(x)(hypothesis) 개선.
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))