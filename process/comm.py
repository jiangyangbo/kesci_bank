#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-6-21 下午9:31
# @Author  : J.W.
# @File    : comm.py

import datetime
import pandas as pd
import pickle
from datetime import date
from numpy import int64

import config
from logger import logger


def split_data_info(data, n):
    '''
    把学生数据分成, 每份 n个
    :param data:
    :param n:
    :return:
    '''
    logger.info("split data, ever item %d students." % n)
    df = pd.DataFrame(columns=data.columns)
    datalist = []

    for i in range(1, len(data) + 1):
        idx = i - 1
        if i % n != 0:
            df = df.append(data.iloc[idx])
        else:
            df = df.append(data.iloc[idx])
            datalist.append(df)
            df = pd.DataFrame(columns=data.columns)
    datalist.append(df)
    return datalist


def accept_this_date(date_time, monthes):
    '''

    :param date_time:
    :param monthes:list of (year,month)
    :return:
    '''
    if date_time == '1/1/0001 00:00:00':
        return False
    if monthes is None:
        return True
    if isinstance(date_time, str):
        try:
            date_time = datetime.datetime.strptime(date_time, '%Y-%m-%d %H:%M:%S')
        except Exception as e:
            logger.error(e, exc_info=True)
            return False
    if isinstance(date_time, datetime.datetime):
        this_year = date_time.year
        this_month = date_time.month

    for accept_date in monthes:
        that_year, that_month = accept_date
        if this_year == that_year and this_month == that_month:
            return True

    return False


def calculate_age(born):
    '''
    出生日期转换成年龄
    :param born:  date类型
    :return:年龄
    '''
    if isinstance(born, datetime):
        born = date(born.year, born.month, born.day)

    if not isinstance(born, date):
        return -1

    today = date.today()
    try:
        birthday = born.replace(year=today.year)
    except ValueError:
        # raised when birth date is February 29
        # and the current year is not a leap year
        birthday = born.replace(year=today.year, day=born.day - 1)
    if birthday > today:
        return today.year - born.year - 1
    else:
        return today.year - born.year


def view_dict(dict_info):
    for k, v in dict_info.items():
        logger.info("%s: %d %s" % (k, len(v), v))
    pass


class GarnetException(Exception):
    def __init__(self, message='Unknown Error！'):
        # self.code = code
        self.msg = message


def save_file(obj, file_path):
    '''
    保存文件
    :param file_path:
    :return:
    '''
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)
    logger.info("Object saved in: %s", file_path)
    return file_path


def load_file(file_path):
    '''
    从文件中加载
    :param file_path:
    :return:
    '''
    with open(file_path, 'rb') as f:
        obj = pickle.load(f)
    logger.info("Loaded file from: %s", file_path)
    return obj


def num_to_str(i):
    '''
    数字转字符串，保证两位编码
    :param i:
    :return:
    '''
    if i > 99 or i < 0:
        raise RuntimeError("Argument i is too big. required [0, 99], given %d " % i)
    if i < 10:
        return '0' + str(i)
    else:
        return str(i)


def feature_extention(features):
    '''
    电话等属性  有或无的特征
    :param features:
    :return:
    '''
    if features is None:
        return features
    # 空的填充为 null
    features = features.fillna('null')
    name = 'contact'
    for name in config.extention_features:
        # 添加新列 null的赋值为1
        try:
            if isinstance(features[name][0], str):
                features.loc[features[name] == 'null', name + '_null'] = '1'
            elif isinstance(features[name][0], int64):
                features.loc[features[name] == 0, name + '_null'] = '1'
        except Exception as e:
            logger.error(e, exc_info=True)
    # 剩余的填充为 0

    features = features.fillna('0')
    if config.DEBUG:
        logger.info(features.head())
        logger.info(features.columns)
    return features


def create_counter():
    '''
    https://blog.csdn.net/sinat_41701878/article/details/79301449
    :return: 每次调用自增1
    '''

    def increase():  # 定义一个含有自然数算法的生成器,使用next来完成不断调用的递增
        n = 0
        while True:
            n = n + 1
            yield n

    it = increase()  # 一定要将生成器转给一个(生成器)对象,才可以完成

    def counter():  # 再定义一内函数
        return next(it)  # 调用生成器的值,每次调用均自增

    return counter


if __name__ == "__main__":
    d = datetime.datetime.strptime('2014-08-15 10:27:36', '%Y-%m-%d %H:%M:%S')
    m = d.month
    d = date(year=2014, month=2, day=1)
    print(d)
    m = []
    m.append((2014, 8))
    m.append((2014, 7))
    r = accept_this_date('2014-08-15 10:27:36', m)
    print(r)
