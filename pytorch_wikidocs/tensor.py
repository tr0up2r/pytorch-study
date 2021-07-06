import numpy as np
import torch

# wikidocs의 PyTorch로 시작하는 딥러닝 입문 문서로 하는 study.
# https://wikidocs.net/52460

# 1D with Numpy
t1 = np.array([0., 1., 2., 3., 4., 5.,6.])
print('t1 :', t1)

# 1차원 vector의 차원(dimension)과 크기(shape)를 출력
print('Rank of t :', t1.ndim)
print('Shape of t : ', t1.shape)

# Numpy 원소 접근 (by indexing)
print('t1[0] t1[1] t1[-1] = ', t1[0], t1[1], t1[-1], '\n')

# slicing - array와 동일... 생략


# 2D with Numpy
t2 = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]])
print('t2 :', t2)

# 2차원 vector의 차원(dimension)과 크기(shape)를 출력
print('Rank of t :', t2.ndim)
print('Shape of t : ', t2.shape, '\n')




# 1D with PyTorch
pt1 = torch.FloatTensor([0., 1., 2., 3., 4., 5., 6.])
print('pt1 :', pt1)

# dim, shape, size 출력해보기 (shape, size() 이용하면 크기를 확인해볼 수 있음)
print(pt1.dim())
print(pt1.shape)
print(pt1.size(), '\n')

# indexing, slicing 생략


# 2D with PyTorch
pt2 = torch.FloatTensor([[1., 2., 3.],
                         [4., 5., 6.],
                         [7., 8., 9.],
                         [10., 11., 12.]
                         ])
print('pt2 :', pt2)

print(pt2.dim())
print(pt2.size())

# Slicing
print(pt2[:, 1])
print(pt2[:, 1].size())
print(pt2[:, :-1], '\n')


# Broadcasting
# PyTorch에서 자동으로 크기를 맞춰 행렬 연산을 수행하게 만드는 기능

# 같은 크기일 때의 연산
m1 = torch.FloatTensor([3, 3])
m2 = torch.FloatTensor([2, 2])
print(m1 + m2)

# 2 x 1 Vector + 1 x 2 Vector
m1 = torch.FloatTensor([1, 2])
m2 = torch.FloatTensor([3], [4])