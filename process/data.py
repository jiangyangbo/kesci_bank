#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-10-23 上午10:38
# @Author  : J.W.
# @File    : data.py
from time import time

import pandas as pd

import config
from logger import logger
from process import comm


def load_csv(path, header=0, names=None):
    '''
    读取csv数据成DataFrame
    :param path: excel文件路径
    :param header: 标题位置
    :param names: 重命名
    :return: DataFrame
    '''
    logger.info("Loading data from: %s", path)
    data = pd.read_csv(path, header=header)
    if names is not None:
        data = data.rename(columns=names)
    logger.info(data.shape)
    view_data(data)
    return data


def load_train_data():
    '''
    :return:
    '''
    return comm.load_file(config.train_df_path)

def load_test_data():
    '''
    :return:
    '''
    return comm.load_file(config.test_df_path)

def view_data(data):
    logger.info("Columns: \n%s", data.columns)
    s = "\n".join([str(c) for c in data.columns])
    print(s)
    import pdb
    #pdb.set_trace()
    if config.date_judgment_flag in data.columns:
        all_date = set()
        for idx in data.index:
            #logger.info("\n index  %s", str(idx))
            row = data.iloc[idx]
            get_time_server = row[config.date_judgment_flag]
            all_date.add(get_time_server[:7])
        all_date = list(all_date)
        all_date.sort()
        logger.info('all_date: {}'.format('\n'.join(all_date)))

    logger.info("\n %s", data.head(2))


def dump_data():
    '''
    读取excel 文件并保存成pickle格式，加快读取速度
    '''
    start = time()
    if config.train_flag:
        sale_data_df = load_csv(config.train_path) #

        end = time()
        logger.info("Use time: %s", str(end - start))
        comm.save_file(sale_data_df, config.train_df_path)
    else:
        sale_data_df = load_csv(config.test_path)  #

        end = time()
        logger.info("Use time: %s", str(end - start))
        comm.save_file(sale_data_df, config.test_df_path)


#import datetime
#import arrow
import numpy as np
from pandas import Series
import pdb
#  添加时间的特征 7 个
#def parse_TimeInfo(date):
#    d = arrow.get(date, "YYYY-MM-DD HH:mm:ss")
#    return Series(d.year,
#                  index=[ 'first_data_year'],
#                  dtype=np.int32)


#def datetime_apply(data):
#    date_year = data['first_get_date'].apply(parse_TimeInfo)
#    return date_year





if __name__ == "__main__":
    dump_data()
