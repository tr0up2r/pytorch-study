import torch
import torch.nn as nn
import torch.nn.functional as F

# PyTorch에서 데이터를 좀 더 쉽게 다룰 수 있도록 제공하는 도구인 dataset과 dataloader.
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

x_train  =  torch.FloatTensor([[73,  80,  75],
                               [93,  88,  93],
                               [89,  91,  90],
                               [96,  98,  100],
                               [73,  66,  70]])
y_train  =  torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])

dataset = TensorDataset(x_train, y_train)

# dataset을 만들었으면 dataloader를 사용 가능.
# dataset과 minibatch의 크기를 기본 인자로 받음.
# shuffle - epoch마다 dataset를 섞을지 여부. (True 권장!)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# model과 optimizer 설계.
model = nn.Linear(3,1)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

nb_epochs = 20
for epoch in range(nb_epochs + 1):
    for batch_idx, samples in enumerate(dataloader):
        print('batch_idx :', batch_idx)
        print('samples :', samples)
        x_train, y_train = samples
        # H(x) 계산
        prediction = model(x_train)

        # cost 계산
        cost = F.mse_loss(prediction, y_train)

        # cost로 H(x) 계산
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}\n'.format(
            epoch, nb_epochs, batch_idx+1, len(dataloader),
            cost.item()
        ))