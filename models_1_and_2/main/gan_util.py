#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
=================================================
@Project -> File   ：classification -> gan
@IDE    ：PyCharm
@Author ：zspp
@Date   ：2021/9/24 15:15
==================================================
"""
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import random
import os
from .util import resource_path
import warnings
import matplotlib.pyplot as plt
import pickle

tf.compat.v1.disable_eager_execution()
tf.get_logger().setLevel('ERROR')

Epoch = 15000
LR_G = 0.0001
LR_D = 0.0001
N_IDEAS = 5
NumOfLine = 16
BATCH_SIZE = 32
out_columns = ['Fluid volume (m3)Exe', 'Proppant/Exe']
gan_scalar_in_path = resource_path('model/scalar/out_scalar.pickle')
gan_scalar_out_path = resource_path('model/scalar/in_scalar.pickle')
gan_model_path = resource_path('model/gan/')
execute_data_path = resource_path('data/execute_data.csv')
Reservoir_data_path = resource_path('data/mean_std_data.csv')
Reservoir_data_class_path = resource_path('data/mean_std__class_data.csv')
view_number = 30


def gan(in_, out_):
    """
    gan模型
    :param in_: 输入储层特征数据，dataframe格式
    :param out_: 输出施工参数数据，dataframe格式
    :return:
    """
    out_columns = out_.columns

    # 设置tensorflow和numpy随机种子数,保证每次随机量相同
    tf.compat.v1.set_random_seed(1)
    np.random.seed(1)

    NumOfF = out_.columns.size  # 软件系统优化参数的数量
    data = in_.iloc[:, 1:].values  # 将数据转换成矩阵
    out_data = out_.values

    # 数据标准化,去均值和方差归一化
    ss = StandardScaler()
    data = ss.fit_transform(data.astype(float))

    s2 = StandardScaler()
    out_data = s2.fit_transform(out_data.astype(float))

    s = pickle.dumps(s2)
    with open(gan_scalar_out_path, 'wb') as f:
        f.write(s)

    s = pickle.dumps(ss)
    with open(gan_scalar_in_path, 'wb') as f:
        f.write(s)

    # 小批量抽取训练样本
    def Cwork():
        clist = random.sample(range(data.shape[0]), NumOfLine)  # 从1～BATCH_SIZE中随机取NumOfLine个数
        dataused = np.zeros(shape=(NumOfLine, data.shape[1]))
        outused = np.zeros(shape=(NumOfLine, NumOfF))
        out_before = []
        j = 0
        for c in clist:
            # 返回某一行   data[i:i+1]
            dataused[j] = data[c]
            outused[j] = out_data[c]
            out_before.append(in_.iloc[c, 0])
            j = j + 1
        return dataused, outused, out_before

    # 重置/清除计算图
    tf.compat.v1.reset_default_graph()

    # 建立一个三层神经网络作为GAN的生成器，输入为随机噪声，输出为NumOfF个参数
    with tf.compat.v1.variable_scope('Generator'):  # 返回一个用于定义创建variable（层）的op的上下文管理器
        # G_in = tf.compat.v1.placeholder(tf.float32, [None, N_IDEAS],
        #                                 name='random_in')  # random ideas (could from normal distribution)
        fearture_in_ = tf.compat.v1.placeholder(tf.float32,
                                                [None, data.shape[1]],
                                                name='fearture_in')  # random ideas (could from normal distribution)

        # G_cat = tf.concat([G_in, fearture_in_], 1)

        G_l1 = tf.compat.v1.layers.dense(fearture_in_, 128, tf.nn.relu)  # 创建一个128个神经元的隐藏层，激励函数为relu
        G_out = tf.compat.v1.layers.dense(G_l1, NumOfF, name='G_OUT')  # 创建一个NumOfF个神经元的输出层
        predict = tf.compat.v1.identity(G_out, name='predict')

    # 建立一个三层神经网络作为GAN的判别器，输入为一组参数，输出为0-1的数
    with tf.compat.v1.variable_scope('Discriminator'):
        real_f = tf.compat.v1.placeholder(tf.float32, [None, NumOfF], name='real_in')  # 训练样本
        real_f_ = tf.concat([real_f, fearture_in_], 1)

        D_l0 = tf.compat.v1.layers.dense(real_f_, 128, tf.nn.relu, name='l')
        p_real = tf.compat.v1.layers.dense(D_l0, 1, tf.nn.sigmoid, name='out')  # 样本来自训练集的可能性
        # 重复使用上面的两层l和out
        G_out_ = tf.concat([predict, fearture_in_], 1)
        D_l1 = tf.compat.v1.layers.dense(G_out_, 128, tf.nn.relu, name='l', reuse=True)  # 生成样本
        p_fake = tf.compat.v1.layers.dense(D_l1, 1, tf.nn.sigmoid, name='out', reuse=True)  # 样本来自训练集的可能性

    # 定义判别器D的损失函数D_loss和生成器G的损失函数G_loss
    D_loss = -tf.reduce_mean(tf.compat.v1.log(p_real) + tf.compat.v1.log(1 - p_fake))
    G_loss = tf.reduce_mean(tf.compat.v1.log(1 - p_fake))

    # 选择AdamOptimizer优化器来最小化损失函数
    train_D = tf.compat.v1.train.RMSPropOptimizer(LR_D).minimize(
        D_loss, var_list=tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator'))
    train_G = tf.compat.v1.train.RMSPropOptimizer(LR_G).minimize(
        G_loss, var_list=tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='Generator'))

    # 开始使用tensorflow训练GAN
    sess = tf.compat.v1.Session()  # 创建tensorflow会话
    sess.run(tf.compat.v1.global_variables_initializer())  # 对上述所有定义的变量进行初始化

    outp = []  # 输出300个生成值
    out_name = []
    i = 0  # 表示第几个生成值
    loss_dict = {'dloss': [], 'gloss': []}
    for step in range(Epoch + 1):  # 进行Epoch次迭代训练
        dataused, outused, out_before = Cwork()  # 多次采集不同数量的训练样本
        # print(out_before)
        # G_ideas = np.random.randn(NumOfLine, N_IDEAS)  # G的输入是NumOfLine个N_IDEAS维随机噪声
        # 训练GAN并得到相应输出
        results, pa0, Dl, Gl = sess.run([predict, p_fake, D_loss, G_loss, train_D, train_G],
                                        {real_f: outused, fearture_in_: dataused})[:4]

        loss_dict['dloss'].append(-Dl)
        loss_dict['gloss'].append(Gl)
        if step % 200 == 0:
            print("判别器损失：", -Dl, "，生成器损失：", Gl, "----step", step)

        if step > (Epoch - view_number):  # 将最后10次生成的结果第一条作为最终生成集
            outp.append(results[0])
            out_name.append(out_before[0])
            i = i + 1

    outp = s2.inverse_transform(outp)  # 将标准化的数据转换成原本格式
    # print(out_name)
    save_ckpt_model(sess, gan_model_path, Epoch)
    return pd.DataFrame(outp, columns=out_columns).dropna(), loss_dict, out_name


def cgan(in_, out_):
    """
    cgan模型训练
    :param in_: dataframe形式，储层特征数据
    :param out_: dataframe形式，施工参数数据
    :return: 训练集测试结果
    """

    out_columns = list(out_.columns).append('Well Name')

    res, loss_dict, out_name = gan(in_, out_)
    res.insert(2, 'Well Name', out_name)
    # res[['Well Name']] = out_name
    # res = res[res.iloc[0]> 0]
    # print(res)
    if res.empty:
        pd.DataFrame(columns=out_columns)

    while True:  # 保证生成想要的样本数量
        if res.shape[0] >= view_number:
            res = res.iloc[:view_number, :]
            break
        else:
            gan_results, loss_dict, out_name = gan(in_, out_)
            gan_results.insert(2, 'Well Name', out_name)
            if not gan_results.empty:
                # gan_results = gan_results[gan_results.iloc[:,:-1]  > 0]
                res = pd.concat([res, gan_results], axis=0)
                res.dropna(axis=0, how='any', inplace=True)
    # res[performance] = np.NAN
    res.reset_index()

    fig = plt.figure()
    plt.plot(loss_dict['dloss'], color='red', label='D')
    plt.plot(loss_dict['gloss'], color='blue', label='G')
    plt.title('loss')
    plt.legend()
    plt.savefig(resource_path('output/gan_loss.png'))

    return res


def save_ckpt_model(sess, save_path, train_step=None):
    """
    保存训练好的模型
    :param sess: 会话对象
    :param save_path: 保存模型的路径
    :param train_step:
    :return:
    """
    saver = tf.compat.v1.train.Saver(max_to_keep=5)
    # 检查路径是否存在
    path = os.path.abspath(save_path)  # 获取绝对路径
    if os.path.exists(path) is False:
        os.makedirs(path)
        print("成功创建模型保存新路径：{}".format(path))
    saver.save(sess, save_path + 'model.ckpt', global_step=train_step, write_meta_graph=True)  # 保存为ckpt模型
    print("成功使用CKPT模式保存模型到路径：{}".format(path))


def load_ckpt_model(sess, save_path):
    """
    加载训练好的模型
    :param sess: 会话对象
    :param save_path: 模型保存的路径
    :return:
    """
    checkpoint = tf.train.get_checkpoint_state(save_path)  # 从checkpoint文件中读取checkpoint对象
    input_checkpoint = checkpoint.model_checkpoint_path
    # print(input_checkpoint)
    input_checkpoint = os.path.join(save_path, input_checkpoint.split('/')[-1])
    saver = tf.compat.v1.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)  # 加载模型结构
    saver.restore(sess, input_checkpoint)  # 使用最新模型
    # sess.run(tf.compat.v1.global_variables_initializer())  # 初始化所有变量


def train_gan():
    # 训练gan模型
    feature_data = pd.read_csv(Reservoir_data_path)

    excute_data = pd.read_csv(execute_data_path)

    all_data = pd.merge(left=feature_data, right=excute_data, on='Well Name')

    in_ = all_data[list(feature_data.columns)]
    out_ = all_data[out_columns]

    res = cgan(in_, out_)

    col_list = []
    for i in list(excute_data.columns):
        if i != 'Well Name':
            i = i + '__TRUE'
        col_list.append(i)
    excute_data.columns = col_list
    res = pd.merge(left=res, right=excute_data, on='Well Name')

    return res


def test(feature_data):
    """
    使用训练好的模型进行测试
    :param feature_data: 储层提取处理过的特征
    :return:  推荐的施工参数，dataframe形式
    """
    # 加载标准化模型
    s_in = pickle.load(open(gan_scalar_in_path, 'rb+'))
    s_out = pickle.load(open(gan_scalar_out_path, 'rb+'))

    # 测试
    sess = tf.compat.v1.Session()
    load_ckpt_model(sess, gan_model_path)
    # 获取计算图节点
    graph = tf.compat.v1.get_default_graph()  # 获取计算图


    y = graph.get_tensor_by_name("Generator/fearture_in:0")
    output = graph.get_tensor_by_name("Generator/predict:0")

    data_in = s_in.transform(feature_data)

    y_pred = sess.run(output, {y: data_in})
    outp = s_out.inverse_transform(y_pred)
    outp = pd.DataFrame(outp, dtype=float)

    outp.columns = out_columns
    outp = outp.round(2)
    return outp
