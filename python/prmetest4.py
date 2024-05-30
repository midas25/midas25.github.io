import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
'''
import matplotlib.pyplot as plt
'''

def determine_true_false(value):
    if value <= 1.5:
        return False
    elif 1.5 < value <= 1.75:
        return np.random.rand() >= 0.95
    elif 1.75 < value <= 2.1:
        return np.random.rand() >= 0.9
    elif 2.1 < value <= 2.5:
        return np.random.rand() >= 0.2
    elif 2.5 < value <= 3.0:
        return np.random.rand() >= 0.1
    else:
        return True  # 100% True for values > 2.5


# Generate 10,000 random values between 1.0 and 5.0 rounded to 1 decimal place
np.random.seed()  # For reproducibility
random_values = np.round(np.random.uniform(0.5, 4.5, 100), 1)


# Apply the function to each value and store the results in a list as [value, result]
results = [[value, determine_true_false(value)] for value in random_values]
x_data = [results[i][0] for i in range(len(results))]
y_data = [results[i][1] for i in range(len(results))]

torch.manual_seed(1)

# 데이터 정의
x_train = torch.FloatTensor(x_data)
x_train = x_train.view([-1, 1])
y_train = torch.FloatTensor(y_data)
y_train = y_train.view([-1, 1])

# 모델 정의
class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.linear = nn.Linear(1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))

model = BinaryClassifier()

# optimizer 설정
optimizer = optim.SGD(model.parameters(), lr=1)
criterion = nn.BCELoss()

# 학습 함수 정의
def train(model, x_train, y_train, optimizer, criterion, nb_epochs):
    for epoch in range(nb_epochs + 1):
        hypothesis = model(x_train)
        cost = criterion(hypothesis, y_train)
        
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch {epoch:4d}/{nb_epochs} Cost: {cost.item():.6f}')

# 학습 실행
nb_epochs = 5000
train(model, x_train, y_train, optimizer, criterion, nb_epochs)

# 학습된 가중치와 바이어스 추출
W = model.linear.weight.data
b = model.linear.bias.data
print(f'Learned parameters:\nW: {W.numpy()}\nb: {b.numpy()}')

'''
# 계산
re_x=5
re_y = round(float(torch.sigmoid(re_x * W + b)), 2)
plt.scatter(re_x, re_y, c='blue', s=100, label='Predicted Point')  # 새로운 점을 표시

# x_train 값 시각화
x_train_np = x_train.numpy()
y_train_np = y_train.numpy()
plt.scatter(x_train_np[:, 0], y_train_np[:], c=y_data, s=100, cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# 시그모이드 함수 시각화 (단일 입력 변수 x에 대해)
x_values = np.linspace(-10, 10, 100)
z = W[0][0].item() * x_values + b.item()
sigmoid = 1 / (1 + np.exp(-z))

plt.plot(x_values, sigmoid, label='Sigmoid Function', markersize=1)
plt.title('Sigmoid Function')
plt.xlabel('x')
plt.ylabel('Sigmoid(x)')
plt.xlim(-3, 9)
plt.legend()
plt.show()
'''