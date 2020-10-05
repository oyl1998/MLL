# -*- coding: utf-8 -*-
'''
Name: decision_tree.py
Auth: long_ouyang
Time: 2020/9/23 17:34
'''

'''
年龄：0代表青年，1代表中年，2代表老年；
有工作：0代表否，1代表是；
有自己的房子：0代表否，1代表是；
信贷情况：0代表一般，1代表好，2代表非常好；
类别(是否给贷款)：no代表否，yes代表是。
'''

from math import log

def creatDataSet():
    '''
    创建数据集
    :return:    dataSet - 数据集
                labels - 分类属性
    '''
    dataSet = [
        [0, 0, 0, 0, 'no'],
        [0, 0, 0, 1, 'no'],
        [0, 1, 0, 1, 'yes'],
        [0, 1, 1, 0, 'yes'],
        [0, 0, 0, 0, 'no'],
        [1, 0, 0, 0, 'no'],
        [1, 0, 0, 1, 'no'],
        [1, 1, 1, 1, 'yes'],
        [1, 0, 1, 2, 'yes'],
        [1, 0, 1, 2, 'yes'],
        [2, 0, 1, 2, 'yes'],
        [2, 0, 1, 1, 'yes'],
        [2, 1, 0, 1, 'yes'],
        [2, 1, 0, 2, 'yes'],
        [2, 0, 0, 0, 'no']
    ]
    labels = ['不放贷', '放贷']
    return dataSet, labels

def calcShannonEnt(dataSet):
    '''
    计算给定数据集的经验熵(香农熵)
    :param dataSet:
    :return: shannonEnt 香农熵
    '''
    numEntires = len(dataSet)
    labelCounts = {}
    for vec in dataSet:
        currentLabel = vec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0

    # 计算香农熵
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntires
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

def splitDataSet(dataSet, axis, value):
    '''
    按照给定的特征划分数据集
    :param dataSet: 待划分的数据集
    :param axis: 划分数据集的特征
    :param value: 需要返回的特征的值
    :return:
    '''
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:]) # 去除axix的特征
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    '''
    选择最优的特征项
    :param dataSet: 数据集
    :return: 信息熵最大的特征的索引值
    '''
    numFeatures = len(dataSet[0]) - 1       # 特征数量
    baseEntropy = calcShannonEnt(dataSet)   # 计算数据集的香农熵
    bestInfoGain = 0.0
    bestFeature = -1        # 最优特征的索引值
    for i in range(numFeatures):
        featlist = [example[i] for example in dataSet]
        uniqueVals = set(featlist)
        newEntropy = 0.0
        # 计算信息增益
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)    # 划分后的子集
            prob = len(subDataSet) / float(len(dataSet))    # 计算子集的概率
            newEntropy += prob * calcShannonEnt(subDataSet) # 计算经验条件熵
        infoGain = baseEntropy -  newEntropy
        print("第%d个特征的增益为%.3f" % (i, infoGain))
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

if __name__ == '__main__':
    dataSet, labels = creatDataSet()
    print('最优特征索引值：' + str(chooseBestFeatureToSplit(dataSet)))
