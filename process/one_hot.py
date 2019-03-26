#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-11-22 下午4:46
# @Author  : J.W.
# @File    : one-hot.py


from sklearn import preprocessing
import numpy as np
from config import Config
from logger import logger
from process import data
from process.feature import collect_clue_features


def one_hot_test():
    enc = preprocessing.OneHotEncoder()  # 创建对象
    enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])  # 拟合
    array = enc.transform([[0, 1, 3]]).toarray()  # 转化
    print(array)

    enc2 = preprocessing.OneHotEncoder()
    enc2.fit([[0], [1], [1], [1]])
    a = enc2.transform([[1]]).toarray()[0]
    b = enc2.transform([[0]]).toarray()
    print(a)
    print(b)

def encode_one_hot(clue_features, columns):
    '''
    类型的 one-hot 编码,
    :param clue_features:data of df
    :param columns:
    :return:
    '''
    encoder = preprocessing.OneHotEncoder()
    data = clue_features[columns]
    data = np.array(data)

    encoder.fit(data)
    try:
        test = encoder.transform([data[0]]).toarray()[0]
        logger.info("Encoder is OK! {}".format(test))
    except Exception as e:
        logger.error(e, exc_info=True)



def encode_org_type(df):
    '''
    线索机构类型 one-hot 编码
    :param df: data of df
    :return:
    '''
    encoder = preprocessing.OneHotEncoder()
    org_type = df['org_type']
    all_types = set(org_type.values)
    types = [[v] for v in all_types]
    logger.info("All org types: {}".format(types))

    encoder.fit(types)
    try:
        test = encoder.transform([types[0]]).toarray()[0]
        logger.info("Encoder is OK! {}".format(test))
    except Exception as e:
        logger.error(e, exc_info=True)
    types = [encoder.transform([[v]]).toarray()[0] for v in df['org_type']]
    df['features'] = np.ndarray(types)
    print(type(df['features']))

    pass


if __name__ == '__main__':
    conf = Config()
    sale_data = data.load_sale_data()
    clue_data = data.load_clue_data()
    logger.info(clue_data.columns)

    clue_features = collect_clue_features(clue_data)
    logger.info(clue_features.shape)
    encode_org_type(clue_features)
    encode_one_hot(clue_features, ['org_type', 'source_type'])
