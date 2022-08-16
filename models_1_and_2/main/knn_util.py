#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
=================================================
@Project -> File   ：classification -> knn_util
@IDE    ：PyCharm
@Author ：zspp
@Date   ：2021/9/24 15:57
==================================================
"""

import math
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .util import resource_path

kmean_pca_path = resource_path('model/knn/pca.pickle')
kmean_scalar_path = resource_path('model/knn/scalar.pickle')
Reservoir_data_path = resource_path('data/mean_std_data.csv')
Reservoir_data_class_path = resource_path('data/mean_std__class_data.csv')
execute_data_path = resource_path('data/execute_data.csv')
kmean_model_path = resource_path('model/knn/kmeans.pickle')
out_columns = ['Fluid volume (m3)Exe', 'Proppant/Exe']

pca_number = 4
K = 5


def knn_train():
    """
    训练聚类模型
    :param Reservoir_data:储层名称+储层特征
    :param K: kmeans参数
    :return:Reservoir_data：储层名称+处理后的特征+类别
    """
    Reservoir_data = pd.read_csv(Reservoir_data_path)
    # print(Reservoir_data[Reservoir_data['Well Name'] == "SN0003-02"].values)
    X = Reservoir_data.iloc[:, 1:].values

    # 降维
    pca = PCA(n_components=pca_number)
    X = pca.fit_transform(X)
    s = pickle.dumps(pca)
    with open(kmean_pca_path, 'wb') as f:
        f.write(s)

    # 标准化
    scalar = StandardScaler()
    X = scalar.fit_transform(X)
    s = pickle.dumps(scalar)
    with open(kmean_scalar_path, 'wb') as f:
        f.write(s)

    model = KMeans(n_clusters=K, random_state=0)
    model.fit(X)

    s = pickle.dumps(model)
    with open(kmean_model_path, 'wb') as f:
        f.write(s)

    y_pre = model.predict(X)
    Reservoir_data['class'] = y_pre

    newReservoir_data = pd.DataFrame(X)
    newReservoir_data['class'] = y_pre
    newReservoir_data.insert(0, Reservoir_data.columns[0], Reservoir_data.iloc[:, 0])
    # print(newReservoir_data)
    newReservoir_data.to_csv(Reservoir_data_class_path, index=False)

    return True


def eucliDist(A, B):
    """
    欧式距离计算
    :param A: list
    :param B: list
    :return: 距离
    """
    return math.sqrt(sum([(a - b) ** 2 for (a, b) in zip(A, B)]))


def pca_scalar_before_test(pca_path, scalar_path, data):
    """
    在测试前进行降维标准化处理
    :param pca_path: 降维模型路径
    :param scalar_path: 标准化模型路径
    :param data: 测试数据
    :return:
    """
    pca = pickle.loads(open(pca_path, 'rb').read())
    scalar = pickle.loads(open(scalar_path, 'rb').read())
    # print(data)
    data = pca.transform(data)
    data = scalar.transform(data)
    return data


def knn_test(feature):
    """
    测试推荐参数
    :param K: 离测试井层最近的储层数目
    :return: 施工参数范围
    :param Reservoir_data: 储层名称+储层特征+类别
    :param execute_data: 储层名称+施工参数+AOF
    :param model: 聚类模型
    :param feature: 测试的储层提取的储层特征
    """

    Reservoir_data = pd.read_csv(Reservoir_data_class_path)
    execute_data = pd.read_csv(execute_data_path)
    model = pickle.load(open(kmean_model_path, 'rb'))
    # 对测试储层特征进行预处理
    feature = pca_scalar_before_test(kmean_pca_path, kmean_scalar_path, feature)
    # print(feature)
    # print(Reservoir_data[Reservoir_data['Well Name']=='SN0045-04'])

    # 对测试井层进行聚类
    predict = model.predict(feature)
    Reservoir_team = Reservoir_data[Reservoir_data['class'] == predict[0]]
    Reservoir_team = Reservoir_team.reset_index()
    Reservoir_team.drop(columns=['index'], inplace=True)
    # print(Reservoir_team)
    # print(feature[0])

    # 距离计算
    distance_list = {}
    for i in range(len(Reservoir_team)):
        distance = eucliDist(list(Reservoir_team.iloc[i, 1:-1]), feature[0])
        distance_list[Reservoir_team.iloc[i, 0]] = distance
        # print(distance_list)
    # print(distance_list['SN0003-02'])

    # 距离排序选择K个
    sort_distance_list = sorted(distance_list.items(), key=lambda d: d[1], reverse=False)
    # print(sort_distance_list)
    # print(sort_distance_list['SN0045-04'])
    config_list = []
    round_name_k = []
    i = 0
    for name, distance in sort_distance_list:
        if name not in execute_data['Well Name'].values:
            continue
        config = list(execute_data[execute_data['Well Name'] == name].iloc[0, :])
        # print(config)
        round_name_k.append(config[0])
        config.pop(0)
        config_list.append(config)
        i += 1
        if i >= K:
            break

    # 推荐施工参数
    config_list = np.array(config_list)

    max_list = np.max(config_list, axis=0)
    min_list = np.min(config_list, axis=0)

    execute_data_range = pd.DataFrame(np.zeros((2, 2)), columns=['min', 'max'], index=out_columns, dtype=float)
    execute_data_range.iloc[0, 0] = min_list[0]
    execute_data_range.iloc[0, 1] = max_list[0]
    execute_data_range.iloc[1, 0] = min_list[1]
    execute_data_range.iloc[1, 1] = max_list[1]
    execute_data_range = execute_data_range.round(2)

    return execute_data_range, round_name_k
