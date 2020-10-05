# -*- coding: utf-8 -*-
'''
Name: Backpropagation.py
Auth: long_ouyang
Time: 2020/9/27 20:12
'''

import torch
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = torch.Tensor([1.0])
'''
Tensor 中 含有 data , grad 两个数据 
'''
w.requires_grad = True
learningrate = 0.01

def forward(x):
    return x * w

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

print('predict (before training)', 4, forward(4).item())

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l = loss(x, y) # l is a tensor
        l.backward()
        print('\tgrad:', x, y, w.grad.item())
        w.data = w.data - learningrate * w.grad.data
        # Tensor 其中的 grad 也是一个Tensor，故需要 grad.data
        w.grad.data.zero_() # 对w.grad.data进行清零
    print('progress;', epoch, l.item())

print('predict (after training)', 4, forward(4).item())