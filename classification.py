# -*- coding: utf-8 -*-
'''
Name: classification.py
Auth: long_ouyang
Time: 2020/10/5 16:46
'''

import torch.nn.functional as F
import torch
import numpy as np
import matplotlib.pyplot as plt

x_data = torch.Tensor([ [1.0], [2.0], [3.0] ])
y_data = torch.Tensor([ [0], [0], [1] ])
#--------------------------------------------------------#
class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = F.sigmoid(self.linear(x))
        return y_pred

model = LogisticRegressionModel()
#---------------------------------------------------------#
criterion = torch.nn.BCELoss(size_average=False)         # 交叉熵 loss function
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) # 优化器
#---------------------------------------------------------#
for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
#--------------------------------------------------------#

x = np.linspace(0, 10, 200)
x_t = torch.Tensor(x).view((200, 1)) # view 转化成 200 * 1 的张量
y_t = model(x_t)
y = y_t.data.numpy()

plt.plot(x, y)
plt.plot([0, 10], [0.5, 0.5], c='r')
plt.xlabel('Hours')
plt.ylabel('Probability of Pass')
plt.grid() # 生成网格
plt.show()