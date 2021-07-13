import torch
import torch.optim as optim

# 랜덤 시드 고정.
torch.manual_seed(1)

# 행렬로 dataset 구현.
x_train  =  torch.FloatTensor([[73,  80,  75],
                               [93,  88,  93],
                               [89,  91,  80],
                               [96,  98,  100],
                               [73,  66,  70]])
y_train  =  torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])

print('x_train의 크기 :', x_train.shape)
print('y_train의 크기 :', y_train.shape, '\n')


# weight와 bias 선언.
# W가 크기 (3 * 1)인 vector가 되도록((5 * 3) x_train과의 곱을 위해).
W = torch.zeros((3, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# optimizer 설정.
optimizer = optim.SGD([W, b], lr=1e-5)

nb_epochs = 20
for epoch in range(nb_epochs + 1):
    # H(x) 계산.
    # 편향 b는 브로드 캐스팅되어 각 샘플에 더해진다.
    hypothesis = x_train.matmul(W) + b

    # cost 계산.
    cost = torch.mean((hypothesis - y_train) ** 2)

    # cost로 H(x) 개선.
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    print('Epoch {:4d}/{} hypothesis: {} Cost: {:.6f}'.format(
        epoch, nb_epochs, hypothesis.squeeze().detach(), cost.item()
    ))

# cost가 3000이었다가 5까지 떨어지는 것을 확인할 수 있음. 0에 가까워짐.