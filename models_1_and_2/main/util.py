#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
=================================================
@Project -> File   ：classification -> util.py
@IDE    ：PyCharm
@Author ：zspp
@Date   ：2021/9/24 15:06
==================================================
"""

import matplotlib
import matplotlib.patches as patches

matplotlib.use('Agg')
import pandas as pd
import os
from .process_data import get_feature_mean_std_list
import matplotlib.pyplot as plt
import six
import streamlit as st
import numpy as np
from xhtml2pdf import pisa
import base64
import sys



def resource_path(relative_path):
    """
    路径处理
    :param relative_path: 相对路径
    :return: 绝对路径
    """
    if getattr(sys, 'frozen', False):  # 是否Bundle Resource
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


columns = ['DEPTH', 'GR_1', 'PHIE_1', 'PHIT_1', 'SWE_1', 'VCL_1']
well_data_feature_path = resource_path('data/mean_std_data.csv')
excute_use_col = ['Well Name', 'Fluid volume (m3)Exe', 'Proppant/Exe']
excute_aof_use_col = ['Well Name', 'Fluid volume (m3)Exe', 'Proppant/Exe', 'AOFkm3/d', 'Formation']

execute_data_path = resource_path('data/execute_data.csv')
excute_aof_data_path = resource_path('data/execute_aof_data.csv')



def get_csv_file_downloader_html(file_path, file_label='File'):
    """
    构造前端可下载文件的超链接
    :param file_path: 文件路径
    :param file_label: 文件名称
    :return: 超链接
    """
    with open(file_path, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(file_path)}">click to download {file_label}</a>'
    return href


def make_train_data(input_dir):
    """
     返回训练集和测试集文件名列表
    :param input_dir:  包含训练井信息的文件夹绝对路径
    :return:
    """
    col1 = [i + '_mean' for i in columns]
    col2 = [i + '_std' for i in columns]
    col2.pop(0)
    col1[0] = 'DEPTH'
    col = col1 + col2
    col.insert(0, 'Well Name')

    data = []
    for r, ds, fs in os.walk(input_dir):
        for d in ds:
            for f in (os.listdir(os.path.join(r, d))):
                if f.split('.')[-1] == 'csv':
                    input_path = os.path.join(os.path.join(r, d), f)
                    statistic = get_feature_mean_std_list(input_path)

                    statistic.insert(0, (f.split('.')[0]).upper())
                    data.append(statistic)
                else:
                    pass
        for f in fs:
            if f.split('.')[-1] == 'csv':
                input_path = os.path.join(r, f)
                statistic = get_feature_mean_std_list(input_path)
                statistic.insert(0, (f.split('.')[0]).upper())
                data.append(statistic)
            else:
                pass
        break

    # print(col)
    # print(data)
    data = pd.DataFrame(data, columns=col)
    data.to_csv(well_data_feature_path, index=False)


def save_excute_data(file_path):
    """
    提取保存施工数据
    :param file_path:
    :return:
    """
    if file_path.split('\\')[-1].split('.')[-1] == 'csv':
        excute_data = pd.read_csv(file_path)
        # print(excute_data)

    elif file_path.split('\\')[-1].split('.')[-1] == 'xlsx':
        excute_data = pd.read_excel(file_path)

        # print(excute_data)

    excute_aof_data = excute_data[excute_aof_use_col]
    # print(excute_data)
    excute_aof_data.to_csv(excute_aof_data_path, index=False)
    excute_data = excute_data[excute_use_col]
    # print(excute_data)
    excute_data.to_csv(execute_data_path, index=False)


def render_mpl_table(data, row_height=0.625, font_size=14,
                     header_color='#40466e', edge_color='w',
                     bbox=None, header_columns=0,
                     ax=None, rowLabels=None, col_width=None, row_colors=None, **kwargs):
    """
    显示图表
    :return: 图表的图片形式
    """
    if bbox is None:
        bbox = [0, 0, 1, 1]
    if row_colors is None:
        row_colors = ['#f1f1f2', 'w']
    if ax is None:
        # print(data.shape[::-1])
        size = (np.array(data.shape[::-1]) + np.array([1, 1])) * np.array([col_width, row_height])
        # print(size)
        fig, ax = plt.subplots(figsize=size, dpi=800)
        # print(size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, rowLabels=rowLabels, cellLoc='center',
                         **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])
    return fig


def visualization(recomend_data, recomend_data_range, round_name_k):
    """
    可视化展示
    :param recomend_data: gan推荐的参数
    :param recomend_data_range:  kmean推荐的参数范围
    :param round_name_k: k个周围井的名称
    :return:  图像
    """

    excute_aof_data = pd.read_csv(excute_aof_data_path)

    # 查询K个临近井层的信息
    k_data_frame = pd.DataFrame(columns=excute_aof_data.columns)
    for i in range(len(excute_aof_data)):
        if excute_aof_data.iloc[i, 0] in round_name_k:
            k_data_frame = pd.concat((k_data_frame, excute_aof_data.iloc[i:i + 1, :]), axis=0)
    k_data_frame = k_data_frame.reset_index(drop=True)

    x, y = excute_aof_data.iloc[:, 1].values, excute_aof_data.iloc[:, 2].values
    xmin = recomend_data_range.iloc[0, 0]
    ymin = recomend_data_range.iloc[1, 0]
    ymax = recomend_data_range.iloc[1, 1]
    xmax = recomend_data_range.iloc[0, 1]

    # 展示周围点的信息
    k_data_frame = k_data_frame.round(2)
    k_data_frame = k_data_frame.sort_values(by=k_data_frame.columns[-2], ascending=False)
    fig2 = render_mpl_table(k_data_frame, header_columns=0, col_width=3.0)

    st.markdown("5.1 Table: History Operation Parameters")
    st.pyplot(fig2)

    fig = plt.figure(figsize=(10, 5), dpi=200)
    # 绘制所有点以及对应的aof大小
    # plt.scatter(x, y, c=excute_aof_data.iloc[:, -1].values, cmap='YlOrRd', edgecolor='none', s=40)

    # 设置坐标轴范围
    plt.xlim(min(x)-10, max(x)+10)
    plt.ylim(min(y)-50, max(y)+50)

    # 绘制范围
    currentAxis = plt.gca()
    rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                             color='red', fill=False)
    currentAxis.add_patch(rect)

    # 绘制聚类方法推荐的k个点
    sc = plt.scatter(k_data_frame.iloc[:, 1].values, k_data_frame.iloc[:, 2], c=k_data_frame.iloc[:, -2], marker='^',
                     cmap=plt.cm.get_cmap('Reds'), linewidths=15)
    plt.colorbar(sc)

    # 绘制坐标名称
    for i in range(len(k_data_frame)):
        plt.annotate(k_data_frame.iloc[i, 0], xy=(k_data_frame.iloc[i, 1], k_data_frame.iloc[i, 2]),
                     xytext=(k_data_frame.iloc[i, 1] - 0.5, k_data_frame.iloc[i, 2] + 0.1), )

    # 绘制cgan方法推荐的1个点
    plt.scatter(recomend_data.iloc[0, 0], recomend_data.iloc[0, 1], marker='p', c='green', linewidths=15)
    plt.annotate('recommend config', xy=(recomend_data.iloc[0, 0], recomend_data.iloc[0, 1]),
                 xytext=(recomend_data.iloc[0, 0] - 0.5, recomend_data.iloc[0, 1] + 0.1), )

    plt.title("Recommended execution data visualization")
    plt.xlabel('{}'.format(excute_use_col[1][:-7] + '[m3]'))
    plt.ylabel('{}'.format(excute_use_col[2][:-4] + " [km3/day]"))
    plt.tight_layout()
    # plt.update_layout(
    #     xaxis_title="Truth", yaxis_title="Forecast", legend_title_text="", height=450, width=800
    # )
    plt.savefig(resource_path('output/draw.png'))
    return fig, k_data_frame


def down_pdf(feature_data, execute_data1, execute_data2, k_data_frame, name):
    """
    下载报告
    :param feature_data: 储层特征数据，dataframe格式
    :param execute_data1: gan推荐的施工数据，dataframe格式
    :param execute_data2: kmeans推荐的施工数据范围，dataframe格式
    :param k_data_frame: kmeans推荐的临近储层相关数据
    :param name: 储层名称
    :return:
    """
    message1 = """
    <html>
    <head >
        <style>
            @font-face {
                font-family: HGFS1_CNKI;
                src: url('../../fonts/HGFS1_CNKI.TTF') format('woff'),
                url('../fonts/HGFS1_CNKI.TTF') format('truetype')
            }
            body {
                font-family: HGFS1_CNKI;

            }
            h1 {
                font-size: 15pt;
            }
            p {
                font-size: 13pt;
                font-family: HGFS1_CNKI;
            }
            table {
                border : 1pt;
                cellpadding: 10pt;
                text-align:center ;
                vertical-align: middle;
                height :20pt;
                font-size: 8pt;

            }
            th {
                background-color: #dedede;
                font-size: 7pt;
            }
            td {
                background-color: #ffffff;
                color: #000;
                text-align:center ;
                vertical-align: middle;
            }   
            
        </style>
    </head>

    <h1 align="center">%s's execution data recommendation report</h1>
    <body>
    <p >1. The characteristic  information of %s is as follows :</p>
    %s
    """ % (name, name, feature_data.to_html(index=False))

    message2 = """

    <p >2. GAN's recommended execution data information is as follows :</p>
    %s
    """ % (execute_data1.to_html(index=False))
    message3 = """
    <p >3. Clustering's  recommended execution data ranges  are as follows :</p>
    %s
    """ % (execute_data2.to_html())
    message4 = """
    <p >4. The information  of neighboring wells recommended by the cluster based on the characteristic information of the well is as follows :</p>
    %s
    """ % (k_data_frame.to_html(index=False))

    message5 = """
    <p >5. The Visualization is as follows :</p>
    <div >
        <img src="%s" width=800  height=350 />
    </div>
    </body>

    </html>
    """ % (resource_path('output/draw.png'))

    message = "".join([message1, message2, message3, message4, message5])

    # data = open(GEN_HTML).read()
    # print(data)
    result = open(resource_path('output/report.pdf'), 'wb')
    pdf = pisa.CreatePDF(message, result)
    result.close()
    # pisa.startViewer('report.pdf')
