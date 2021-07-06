# 신경망 학습 시 가장 자주 사용되는 알고리즘인 역전파(backpropagation)
# 해당 알고리즘에서 매개변수는 주어진 매개변수에 대한 손실함수의 gradient에 따라 조정됨.
# gradient를 계산하기 위해 torch.autograd라는 자동 미분 엔진이 내장

import torch

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output

w = torch.randn(5, 3, requires_grad=True) # 최적화 해야하는 매개변수(weight)
b = torch.randn(3, requires_grad=True) # 최적화 해야하는 매개변수(bias)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

print('Gradient function for z =', z.grad_fn)
print('Gradient function for loss =', loss.grad_fn, '\n')


# Gradient 계산하기
# weight를 최적화하려면 매개변수에 대한 loss의 도함수를 계산해야 함
loss.backward()
print(w.grad)
print(b.grad, '\n')