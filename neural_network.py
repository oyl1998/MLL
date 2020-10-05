# -*- coding: utf-8 -*-
'''
Name: neural_network.py
Auth: long_ouyang
Time: 2020/9/24 14:21
'''

from sklearn.datasets import load_boston # 波士顿房价
from sklearn import preprocessing
import numpy as np

X, y = load_boston(return_X_y=True)
print(X.shape)
print(y.shape)
X = preprocessing.scale(X[:100, :])
y = preprocessing.scale(y[:100].reshape(-1, 1))
