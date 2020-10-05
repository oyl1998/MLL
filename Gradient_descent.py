# -*- coding: utf-8 -*-
'''
Name: Gradient_descent.py
Auth: long_ouyang
Time: 2020/9/24 19:28
'''

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0
learningRate = 0.01

def forward(x):
    return x * w

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

'''
def gradient(x, y):
    return 2 * x * (x * w - y) # x * w = forward(x)

# every epoch loss
'''

def cost(xs, ys):
    cost = 0.0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        cost += (y_pred - y) * (y_pred - y)
    return cost / len(ys)

# d cost / d w
def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (x * w - y)
        # grad += x * (x * w - y)
    return grad / len(xs)

print('Predict (before training)', 4, forward(4))

import matplotlib.pyplot as plt
cost_total = []
for epoch in range(100):
    cost_val = cost(x_data, y_data)
    cost_total.append(cost_val)
    grad_val = gradient(x_data, y_data)
    w -= learningRate * grad_val
    print('Epoch:', epoch, 'w=', w, 'loss=', cost_val)
print('Pradict (after training)', 4, forward(4))

plt.plot(cost_total)
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.show()
