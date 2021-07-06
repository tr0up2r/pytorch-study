import torch
import numpy as np

# 파이토치 한국 사용자 모임 사이트의 코드를 이용한 study.
# https://tutorials.pytorch.kr/beginner/basics/tensorqs_tutorial.html

# data initialization
data = [[1, 2], [3, 4]]


# 데이터로부터 직접 Tensor 생성하기
# 자료형은 자동으로 유추.
x_data = torch.tensor(data)
print(f"Tensor from Data : \n {x_data} \n")


# NumPy 배열로부터 Tensor 생성하기
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(f"Tensor from NumPy : \n {x_np} \n")


# 다른 Tensor로부터 Tensor 생성하기
# 명시적으로 override하지 않는다면, Tensor의 속성(shape, datatype)을 유지
x_ones = torch.ones_like(x_data) # x_data의 속성을 유지
print(f"Ones Tensor : \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # x_data의 속성을 덮어씀
print(f"Random Tensor: \n {x_rand} \n")


# shape - Tensor의 차원(dimension)을 나타내는 tuple.
shape = (2, 3, )

# 이렇게 해주면 출력 tensor의 dimension을 결정해준다.
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}\n")


# Tensor의 속성(Attribute)
tensor = torch.rand(3, 4)

# Tensor의 모양(shape)
print(f"Shape of tensor: {tensor.shape}")

# Tensor의 자료형(datatype)
print(f"Datatype of tensor: {tensor.dtype}")

# Tensor가 어느 장치에 저장되는지
print(f"Device tensor is stored on: {tensor.device}\n")



# Tensor Operation

# GPU가 있을 경우, Tensor를 GPU로 이동시킬 수 있음
# 본 환경에서는 if문이 false
if torch.cuda.is_available():
    tensor = tensor.to('cuda')

tensor = torch.ones(4, 4)

# indexing & slicing
print('First row: ',tensor[0])
print('First column: ', tensor[:, 0])
print('Last column:', tensor[..., -1])
tensor[:,1] = 0
print(tensor, '\n')


# Tensor 합치기
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1, '\n')


# Arithmetic operations

# 두 Tensor 간 행렬 곱(matrix multiplication) 계산
# 모두 같은 결과 값
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)

# 요소별 곱(element-wise product) 계산.
# 모두 같은 결과 값
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

# single-element Tensor의 경우,
# item() 을 사용하여 python 숫자 값으로 변환 가능
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item), '\n')



# NumPy 변환
# CPU 상, NumPy 배열과 Tensor는 메모리 공간 공유
# 따라서, 하나를 변경하면 다른 하나도 변경
t = torch.ones(5)
print(f't: {t}')
n = t.numpy()
print(f'n: {n}\n')

# in-place 연산
# 연산 결과를 operand(여기에서는 t)에 저장하는 연산
# derivative 계산에 문제 발생 가능하여, 사용 권장 X
t.add_(1)

print(f'after in-place add operation\nt: {t}')
print(f'n: {n}\n')