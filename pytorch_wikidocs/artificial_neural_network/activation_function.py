import numpy as np
import matplotlib.pyplot as plt


# 시그모이드 함수 그래프를 그리는 코드.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)

plt.plot(x, y)
plt.plot([0, 0], [1.0, 0.0], ':')  # 가운데 점선 추가
plt.title('Sigmoid Function')
plt.show()


# Hyperbolic tangent(tanh) 함수 그래프를 그리는 코드.
x = np.arange(-5.0, 5.0, 0.1) # -5.0부터 5.0까지 0.1 간격 생성
y = np.tanh(x)

plt.plot(x, y)
plt.plot([0,0],[1.0,-1.0], ':')
plt.axhline(y=0, color='orange', linestyle='--')
plt.title('Tanh Function')
plt.show()


# ReLU 함수 그래프를 그리는 코드.
def relu(x):
    return np.maximum(0, x)


x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)

plt.plot(x, y)
plt.plot([0,0],[5.0,0.0], ':')
plt.title('Relu Function')
plt.show()


a = 0.1


# leaky ReLU 함수 그래프를 그리는 코드.
def leaky_relu(x):
    return np.maximum(a*x, x)


x = np.arange(-5.0, 5.0, 0.1)
y = leaky_relu(x)

plt.plot(x, y)
plt.plot([0,0],[5.0,0.0], ':')
plt.title('Leaky ReLU Function')
plt.show()