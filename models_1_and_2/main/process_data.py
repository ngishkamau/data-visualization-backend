#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
=================================================
@Project -> File   ：classification -> data_process
@IDE    ：PyCharm
@Author ：zspp
@Date   ：2021/9/24 15:14
==================================================
"""
import pandas as pd

columns = ['DEPTH', 'GR_1', 'PHIE_1', 'PHIT_1', 'SWE_1', 'VCL_1']


def get_feature_mean_std(input_file):
    """
    获取储层统计相关特征
    :rtype: object
    """
    csv = pd.read_csv(input_file, usecols=columns).iloc[1:].astype(float)
    csv_describe = csv.describe()
    feature = list(csv_describe.iloc[1, :])
    feature[0] = csv_describe.iloc[0, 0]
    feature.extend(list(csv_describe.iloc[2, 1:]))

    col1 = [i + '_mean' for i in columns]
    col2 = [i + '_std' for i in columns]
    col2.pop(0)
    col1[0] = 'DEPTH'
    col = col1 + col2
    # print(feature)
    data = pd.DataFrame([feature], columns=col)
    data = data.round(4)
    return data


def get_feature_mean_std_list(input_file):
    """
    获取储层统计相关特征
    :rtype: object
    """
    csv = pd.read_csv(input_file, usecols=columns).iloc[1:].astype(float)
    csv_describe = csv.describe()
    feature = list(csv_describe.iloc[1, :])
    feature[0] = csv_describe.iloc[0, 0]
    feature.extend(list(csv_describe.iloc[2, 1:]))

    # print(feature)
    return feature


