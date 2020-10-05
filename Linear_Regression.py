# -*- coding: utf-8 -*-
'''
Name: Linear_Regression.py
Auth: long_ouyang
Time: 2020/10/5 15:00
'''

import torch

x_data = torch.Tensor([ [1.0], [2.0], [3.0] ])
y_data = torch.Tensor([ [2.0], [4.0], [6.0] ])

class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    # forward is __call__() function
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

model = LinearModel()

criterion = torch.nn.MSELoss(size_average=False)
# 优化器 在进行梯度更新的时候用learningrate进行优化
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
'''
torch.optim.Adagrad()
torch.optim.Adam()
torch.optim.Adamax()
torch.optim.ASGD()
torch.optim.LBFGS()
torch.optim.RMSprop()
torch.optim.Rprop()
torch.optim.SGD()
'''

for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print('w=', model.linear.weight.item())
print('b=', model.linear.bias.item())

x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print('y_pred=', y_test.data)
