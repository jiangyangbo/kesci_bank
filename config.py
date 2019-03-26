#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-11-7 下午5:07
# @Author  : J.W.
# @File    : config.py

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, 'data')
resource_path = os.path.join(BASE_DIR, 'resource')

LOG_DIR = os.path.join(BASE_DIR, "logs")

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)  # 创建路径

log_file = os.path.join(LOG_DIR, "app.log")

server = False
train_flag = True
train_flag = False
date_judgment_flag = False
date_judgment_flag = False
DEBUG = False
DEBUG = True
# 训练数据
train_path_v1 = "../data/train_set.csv"  #

# 测试数据
test_path_v1 = "../data/test_set.csv"  #

# 测试结果
test_output_path = "./data/test_out.csv"  #


train_df_path_v1 = os.path.join(resource_path, 'train_df_data.pkl')
test_df_path_v1 = os.path.join(resource_path, 'test_df_data.pkl')

# train
train_df_path = train_df_path_v1
train_path = train_path_v1

# test
test_df_path = test_df_path_v1
test_path = test_path_v1

version = '20190308'
train_feature_path = os.path.join(resource_path, ".train.features." + version + ".pkl")
test_feature_path = os.path.join(resource_path, ".test.features." + version + ".pkl")

model_path = os.path.join(resource_path, "model." + version + ".pkl")

"""
Index(['ID', 'age', 'job', 'marital', 'education', 'default', 'balance',
       'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign',
       'pdays', 'previous', 'poutcome', 'y'],
      dtype='object')
      NO	字段名称	数据类型	字段描述
1	ID	Int	客户唯一标识
2	age	Int	客户年龄
3	job	String	客户的职业
4	marital	String	婚姻状况
5	education	String	受教育水平
6	default	String	是否有违约记录
7	balance	Int	每年账户的平均余额
8	housing	String	是否有住房贷款
9	loan	String	是否有个人贷款
10	contact	String	与客户联系的沟通方式
11	day	Int	最后一次联系的时间（几号）
12	month	String	最后一次联系的时间（月份）
13	duration	Int	最后一次联系的交流时长
14	campaign	Int	在本次活动中，与该客户交流过的次数
15	pdays	Int	距离上次活动最后一次联系该客户，过去了多久（999表示没有联系过）
16	previous	Int	在本次活动之前，与该客户交流过的次数
17	poutcome	String	上一次活动的结果
18	y	Int	预测客户是否会订购定期存款业务
"""

train_accept_feature = ['ID', 'age', 'job', 'marital','education','default', 'balance','housing','loan','contact','day',
                         'month','duration', 'campaign', 'pdays', 'previous','poutcome', 'y']

test_accept_feature = ['ID', 'age', 'job', 'marital','education','default', 'balance','housing','loan','contact','day',
                         'month','duration', 'campaign', 'pdays', 'previous','poutcome']
# 缺少
#train_accept_feature = [ 'job', 'marital','education','default', 'balance','housing','loan','contact','month',
#                         'campaign', 'poutcome', 'y']

# 基础特征的扩展 , int型 特征
extention_features = []
# extention_features.append('age')
# extention_features.append('balance')
# extention_features.append('day')
# extention_features.append('duration')
# extention_features.append('campaign')
#
# #extention_features.append('campaign_cleaned')
# #extention_features.append('pdays')
# extention_features.append('previous')
# #extention_features.append('previous_cleaned')

if train_flag:
    extention_features.append('y')

# string 型特征
all_need_coder_columns = ['job', 'marital','education','default', 'housing','loan','contact','month',
                          'poutcome' ]


class Config():
    '''
    配置文件
    '''

    def __init__(self):
        self.train_monthes = []  # 训练数据范围
        self.test_monthes = []  # 测试数据范围
        self.init_accept_date()

    def init_accept_date(self):
        '''
        初始化实验数据范围
        :return:
        '''
        self.train_monthes.append((2018, 2))
        self.train_monthes.append((2018, 3))
        self.train_monthes.append((2018, 4))
        self.train_monthes.append((2018, 5))
        self.train_monthes.append((2018, 6))
        self.train_monthes.append((2018, 7))
        self.train_monthes.append((2018, 8))
        self.train_monthes.append((2018, 9))

        self.test_monthes.append((2018, 10))
        self.test_monthes.append((2018, 11))
