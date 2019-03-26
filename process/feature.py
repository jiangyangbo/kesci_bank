#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-10-23 上午10:08
# @Author  : J.W.
# @File    : feature.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import os
from sklearn.ensemble import RandomForestClassifier
import config
from logger import logger
from process import comm
from process import data

import pdb

def collect_feature(data):
    '''
    提取特征
    :param data:
    :param features:
    :return:
    '''
    if config.train_flag:
        accept_feature = config.train_accept_feature
    else:
        accept_feature = config.test_accept_feature
    result = data[accept_feature]

    return result


from process.str_utils import get_type_by_name


def add_org_type_by_name(data, name='org_name'):
    '''
    根据机构名字对结构类型简单分类
    例如：跆拳道 舞蹈（舞） 英语 艺术 咨询
    :param data:
    :param name:
    :return:
    '''
    data['org_type_name'] = data.apply(lambda x: get_type_by_name(x[name]), axis=1)


def extract_sale_feature(row, feature):
    '''
    @param data: one person information
    @param feature: feature set
    @return:
    '''

    feature.append(row['Level'])
    feature.append(row['Department'])
    feature.append(row['State'])
    feature.append(row['Sex'])

    # extract_age_feature(row, feature)
    #TODO age为0的时候
    feature.append(row['Age'])


def extract_age_feature(row, feature):
    '''
    年龄
    :param row:
    :param feature:
    :return:
    '''
    birthday = row['Birthday']
    age = comm.calculate_age(birthday)
    age2 = row["Age"]
    feature.append(age)


def collect_clue_features(data, monthes=None):
    '''
    线索特征
    :param data:
    :param monthes: 接受的日期
    :return:
    '''
    if not isinstance(data, pd.DataFrame):
        raise comm.GarnetException("data must be DataFrame, now is {}".format(type(data)))

    # accept_feature = ['org_id', 'org_type', 'intention', 'try_type', 'source_type',
    #                   'message_type', 'student_num', 'institution_open_date',
    #                   'create_date', 'campus_num', 'salesman_id', 'deal_type', 'contact', 'contact_tel']
    accept_feature = data.columns.tolist()
    import pdb
    features = []
    logger.info("Extracting clue feature...")
    features_df = pd.DataFrame(features, columns=accept_feature)
    if monthes is None:
        logger.info("Accepet all date.")
        return data
    else:
        logger.info("Accpet date: {}".format(monthes))
        accept = False
        for idx in data.index:
            row = data.iloc[idx]

            get_time_server = row[config.date_judgment_flag]
            accept = comm.accept_this_date(get_time_server, monthes)
            if accept:
                features_df = features_df.append(row)

    #  根据机构名称判断类别，跆拳道、舞蹈等
    add_org_type_by_name(features_df)

    # logger.info(features)
    if config.DEBUG:
        logger.info("\n{}".format(features_df.head(2)))
        logger.info("Shape of clue features: {}".format(features_df.shape))

    return features_df


def time_apply(data):
    data['clue_times'] = data['org_name'].count()
    # if data['deal_type'] == 1:
    #     data['clue_times'] = 1
    return data

def collect_deal_times_features(data):
    '''
       线索特征, 交易成功前处理次数
       :param data:
       :param
       :return:
       '''
    data = data.groupby('org_id').apply(time_apply)
    data.loc[data['deal_type'] == 1, 'clue_times'] = 1  # 由于数据不是很精确，没有成功的，直接指定为１
    return data

def org_apply(data):
    data['org_name_len'] = data['org_name'].count()

def collect_org_name_length_features(data):
    '''
          线索特征, 机构名称长度
          :param data:
          :param
          :return:
          '''
    #data = data.groupby('org_id').apply(org_apply)
    data['org_name_len'] = data['org_name'].str.len()
    data['org_name_len_less_5'] = 0
    data['org_name_len_big_5'] = 0
    #pdb.set_trace()
    #data['org_name_len_less_5'][data['org_name_len '] < 5] = 1  #  < 5
    #data['org_name_len_big_5'][data['org_name_len '] >= 5] = 1  # >= 5

    data['org_name_len_less_5'] = data['org_name_len_less_5'].where(data['org_name_len'] < 5, 1)
    data['org_name_len_big_5'] = data['org_name_len_big_5'].where(data['org_name_len'] >= 5, 1)
    #pdb.set_trace()
    return data

region_northeast = ["黑龙江省", "吉林省", "辽宁省"]
region_northchina = ["北京市", "天津市", "河北省", "山西省","内蒙古自治区", "河南省", "山东省"]
region_centralchina = ["安徽省", "湖北省",  "湖南省", "江西省"]
region_eastchina = ["上海市","江苏省","浙江省", "福建省" ,"山东省"]
region_southchina = ["广东省", "广西壮族自治区","海南省"]
region_southwest = ["四川省", "贵州省","云南省", "重庆市"]
region_northwest = ["陕西省", "甘肃省", "青海省",  "新疆维吾尔自治区",  "宁夏回族自治区", "西藏自治区"]

def collect_clue_region_features(data):
    data['region_northeast'] = data['province'].apply(lambda x: True if x in region_northeast else False)
    data['region_northchina'] = data['province'].apply(lambda x: True if x in region_northchina else False)
    data['region_centralchina'] = data['province'].apply(lambda x: True if x in region_centralchina else False)
    data['region_eastchina'] = data['province'].apply(lambda x: True if x in region_eastchina else False)
    data['region_southchina'] = data['province'].apply(lambda x: True if x in region_southchina else False)
    data['region_southwest'] = data['province'].apply(lambda x: True if x in region_southwest else False)
    data['region_northwest'] = data['province'].apply(lambda x: True if x in region_northwest else False)

    return data

def collect_salesman_region_features(data):
    data['sale_region_northeast'] = data['Province'].apply(lambda x: True if x in region_northeast else False)
    data['sale_region_northchina'] = data['Province'].apply(lambda x: True if x in region_northchina else False)
    data['sale_region_centralchina'] = data['Province'].apply(lambda x: True if x in region_centralchina else False)
    data['sale_region_eastchina'] = data['Province'].apply(lambda x: True if x in region_eastchina else False)
    data['sale_region_southchina'] = data['Province'].apply(lambda x: True if x in region_southchina else False)
    data['sale_region_southwest'] = data['Province'].apply(lambda x: True if x in region_southwest else False)
    data['sale_region_northwest'] = data['Province'].apply(lambda x: True if x in region_northwest else False)

    return data


def min_max_scaler(train, test):
    logger.info("min_max_scaler")
    train = pd.nan_to_num(train)
    test = pd.nan_to_num(test)
    scaler = MinMaxScaler().fit(train)
    stat_train = scaler.transform(train)
    stat_test = scaler.transform(test)
    return stat_train, stat_test


# http://blog.csdn.net/u012102306/article/details/51940147
# 标准化（Standardization or Mean Removal and Variance Scaling)
# 实际应用中，需要做特征标准化的常见情景：SVM
def standard_scaler(train, test):
    logger.info("standard_scaler")
    train = pd.nan_to_num(train)
    test = pd.nan_to_num(test)
    scaler = StandardScaler().fit(train)
    stat_train = scaler.transform(train)
    stat_test = scaler.transform(test)
    return stat_train, stat_test


def clue_salesman_merge_loop(clue_features, salesman_features):
    '''
    单条线索与销售人员的合并
    :param clue_features:
    :param salesman_features:
    :return:
    '''

    logger.info('clue_salesman_combination start.')
    df = pd.DataFrame(columns=clue_features.columns)
    row = clue_features.loc[0]
    for i in salesman_features.index:
        df = df.append(row, ignore_index=True)
    result = pd.concat([df, salesman_features], axis=1)
    logger.info('clue_salesman_combination end.')
    return result


def dorp_df_key(df, name):
    '''
    从打分中删除给定的列
    :param df:
    :param name:
    :return:
    '''
    try:
        if name in df.columns:
            df.drop([name], axis=1, inplace=True)
        else:
            logger.warn('salesman_features columns: {}'.format(df.columns))
    except Exception as e:
        logger.info("columns: {}".format(df.columns))
        logger.error(e, exc_info=True)


def clue_salesman_merge(clue_features, salesman_features):
    '''
    单条线索与销售人员的合并
    :param clue_features:
    :param salesman_features:
    :return:
    '''
    merge_key = 'merge_key'
    # logger.info('clue_salesman_combination start.')
    clue_features[merge_key] = 0
    salesman_features[merge_key] = 0
    result = pd.merge(clue_features, salesman_features, on=merge_key)

    dorp_df_key(result, merge_key)
    dorp_df_key(clue_features, merge_key)
    dorp_df_key(salesman_features, merge_key)

    # logger.info('clue_salesman_combination end.')
    return result


def feature_combination(salesman_features, clue_features, train=True):
    logger.info("salesman_features shape: {}".format(salesman_features.shape))
    logger.info("clue_features shape: {}".format(clue_features.shape))

    salesman_ids = set(salesman_features.Id.tolist())
    clue_salesman_ids = set(clue_features.salesman_id.tolist())

    if train:
        logger.info("Salesman id: {} {}".format(len(salesman_ids), sorted(salesman_ids)))
        logger.info("Clue salesman id: {} {}".format(len(clue_salesman_ids), sorted(clue_salesman_ids)))
        accept_ids = salesman_ids.intersection(clue_salesman_ids)
        logger.info("accept_ids id: {} {}".format(len(accept_ids), accept_ids))
    else:
        accept_ids = clue_salesman_ids
    clue_features = clue_features.loc[clue_features.salesman_id.isin(accept_ids)]
    columns = list(salesman_features.columns)
    columns.extend(list(clue_features.columns))

    logger.info("feature merging...")
    if train:
        # 实验过程
        salesman_features['salesman_id'] = salesman_features['Id']
        # 类型转换？  如果没有会出现下面错误
        # You are trying to merge on object and int64 columns. If you wish to proceed you should use pd.concat
        clue_features['salesman_id'] = clue_features['salesman_id'].apply(int)
        salesman_features['salesman_id'] = salesman_features['salesman_id'].apply(int)
        df = pd.merge(clue_features, salesman_features, on='salesman_id')

    else:
        # 单条预测过程
        assert len(clue_features) == 1
        df = clue_salesman_merge(clue_features, salesman_features)

    logger.info("final feature shape {}".format(df.shape))
    return df


def view_clue_data(clue_data):
    '''
    查看线索数据
    (['org_id', 'org_name', 'org_type', 'intention', 'try_type',
       'source_type', '来源类型.1', '介绍人/机构/会议', '省', '市', '行政区域', '地址', '校长名称',
       'master_tel', 'contact', 'contact_tel', 'contact_title', 'create_date',
       'campus_num', 'student_num', 'first_get_date', 'source_device',
       'channel', 'introducer', '掉档时间', 'last_contact_date', 'message_type',
       'recycle_bin', 'intent_version', 'creator', 'release_times',
       'institution_open_date', 'salesman_id', 'deal_type'],
      dtype='object')
    :param clue_data:
    :return:

    '''
    logger.info("\n {}".format(clue_data.columns))
    logger.info("try_type \n {}".format(clue_data['try_type'].value_counts()))
    logger.info("source_type \n{}".format(clue_data['source_type'].value_counts()))


def encode_edu_attrs(data):
    values = ["unknown", "primary", "secondary", "tertiary"]
    levels = range(0, len(values))
    dict_levels = dict(zip(values, levels))
    for v in values:
        data.loc[data['education'] == v, 'education'] = dict_levels[v]
    return data


def encode_job_attrs(data):
    values = ["unknown","admin.", "management", "technician", "blue-collar", "retired", "services", "student", "unemployed",
              "self-employed", "entrepreneur", "housemaid"]
    levels = range(0, len(values))
    dict_levels = dict(zip(values, levels))
    for v in values:
        data.loc[data['job'] == v, 'job'] = dict_levels[v]
    return data


def encode_marital_attrs(data):
    values = ["unknown", "single", "married"]
    levels = range(0, len(values))
    dict_levels = dict(zip(values, levels))
    for v in values:
        data.loc[data['marital'] == v, 'marital'] = dict_levels[v]
    return data


def encode_marital_attrs(data):
    values = ["unknown", "single", "divorced", "married"]
    levels = range(0, len(values))
    dict_levels = dict(zip(values, levels))
    for v in values:
        data.loc[data['marital'] == v, 'marital'] = dict_levels[v]
    return data


# ['job', 'marital', 'education', 'contact', 'month', 'poutcome']
def encode_contact_attrs(data):
    values = ["unknown", "cellular", "telephone"]
    levels = range(0, len(values))
    dict_levels = dict(zip(values, levels))
    for v in values:
        data.loc[data['contact'] == v, 'contact'] = dict_levels[v]
    return data


# def encode_month_attrs(data):
#     values = ["unknown", "single", "married"]
#     levels = range(0,len(values))
#     dict_levels = dict(zip(values, levels))
#     for v in values:
#         data.loc[data['month'] == v, 'month'] = dict_levels[v]
#     return data

def encode_poutcome_attrs(data):
    values = ["unknown", "failure", "other", "success"]
    levels = range(0, len(values))
    dict_levels = dict(zip(values, levels))
    for v in values:
        data.loc[data['poutcome'] == v, 'poutcome'] = dict_levels[v]
    return data


def trans_num_attrs(data, numeric_attrs):
    bining_num = 10
    bining_attr = 'age'
    data[bining_attr] = pd.qcut(data[bining_attr], bining_num)
    data[bining_attr] = pd.factorize(data[bining_attr])[0] + 1

    # for i in numeric_attrs:
    #     scaler = preprocessing.StandardScaler()
    #     data[i] = scaler.fit_transform(data[i])
    return data


def get_dummy_from_bool(row, column_name):
    ''' Returns 0 if value in column_name is no, returns 1 if value in column_name is yes'''
    return 1 if row[column_name] == 'yes' else 0


def get_correct_values(row, column_name, threshold, df):
    ''' Returns mean value if value in column_name is above threshold'''
    if row[column_name] <= threshold:
        return row[column_name]
    else:
        mean = df[df[column_name] <= threshold][column_name].mean()
        #pdb.set_trace()
        return mean


def test_clean_data(df):
    '''
    INPUT
    df - pandas dataframe containing bank marketing campaign dataset

    OUTPUT
    df - cleaned dataset:
    1. columns with 'yes' and 'no' values are converted into boolean variables;
    2. categorical columns are converted into dummy variables;
    3. drop irrelevant columns.
    4. impute incorrect values
    '''

    cleaned_df = df.copy()
    #pdb.set_trace()
    # convert columns containing 'yes' and 'no' values to boolean variables and drop original columns
    bool_columns = ['default', 'housing', 'loan']
    for bool_col in bool_columns:
        #cleaned_df[bool_col + '_bool'] = df.apply(lambda row: get_dummy_from_bool(row, bool_col)) # , axis=1
        if df[bool_col]  == 'yes':
            cleaned_df[bool_col + '_bool'] = 1
        else:
            cleaned_df[bool_col + '_bool'] = 0

    cleaned_df = cleaned_df.drop(bool_columns)

    # convert categorical columns to dummies

    # drop irrelevant columns
    cleaned_df = cleaned_df.drop(['pdays'])

    # impute incorrect values and drop original columns
    #cleaned_df['campaign_cleaned'] = df.apply(lambda row: get_correct_values(row, 'campaign', 34, cleaned_df), axis=1)
    #cleaned_df['previous_cleaned'] = df.apply(lambda row: get_correct_values(row, 'previous', 34, cleaned_df), axis=1)
    if df['campaign'] > 34:
        cleaned_df['campaign_cleaned'] = 2.75
    else:
        cleaned_df['campaign_cleaned'] = df['campaign']

    # pdb.set_trace()
    if df['previous'] > 34:
        cleaned_df['previous_cleaned'] = 0.9373
    else:
        cleaned_df['previous_cleaned'] = df['previous']

    cleaned_df = cleaned_df.drop(['campaign', 'previous'])

    # ['job', 'marital', 'education', 'contact', 'month', 'poutcome']
    cat_columns = ['job', 'education', 'contact','poutcome'] # marital
    cat_columns = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']


    
    forest = RandomForestClassifier(n_estimators=100)
    # 通过模型预测　unknown
    pdb.set_trace()
    for i in cat_columns:  # ['poutcome', 'education', 'contact']
         print(cleaned_df[i])
         if cleaned_df[i] == 'unknown':
            #test_data = cleaned_df[cleaned_df[i] == 0]
            testX = cleaned_df.drop(cat_columns)
            testX = cleaned_df.drop('ID')
            testX = np.array(testX).reshape(1,-1)
            pdb.set_trace()
            #forest = RandomForestClassifier(n_estimators=100)
            forest = comm.load_file(os.path.join(config.resource_path, i + ".features" + ".pkl"))
            test_predictY = forest.predict(testX).astype(int)
            pdb.set_trace()
            cleaned_df[i] = test_predictY 


    #pdb.set_trace()
    cat_columns_all = ['job_admin.',
       'job_blue-collar', 'job_entrepreneur', 'job_housemaid',
       'job_management', 'job_retired', 'job_self-employed', 'job_services',
       'job_student', 'job_technician', 'job_unemployed', 'job_unknown',
       'marital_divorced', 'marital_married', 'marital_single',
       'education_primary', 'education_secondary', 'education_tertiary',
       'education_unknown', 'contact_cellular', 'contact_telephone',
       'contact_unknown', 'month_apr', 'month_aug', 'month_dec', 'month_feb',
       'month_jan', 'month_jul', 'month_jun', 'month_mar', 'month_may',
       'month_nov', 'month_oct', 'month_sep', 'poutcome_failure',
       'poutcome_other', 'poutcome_success', 'poutcome_unknown']
    for col in cat_columns_all:
        cleaned_df[col] = 0
    ############################################
    #pdb.set_trace()
    for col in cat_columns:
        col_val = df[col]
        cleaned_df[col+'_'+col_val] = 1
    #pdb.set_trace()
    cleaned_df = cleaned_df.drop(cat_columns)
    return cleaned_df



def clean_data(df):
    '''
    INPUT
    df - pandas dataframe containing bank marketing campaign dataset

    OUTPUT
    df - cleaned dataset:
    1. columns with 'yes' and 'no' values are converted into boolean variables;
    2. categorical columns are converted into dummy variables;
    3. drop irrelevant columns.
    4. impute incorrect values
    '''

    cleaned_df = df.copy()
    #pdb.set_trace()
    # convert columns containing 'yes' and 'no' values to boolean variables and drop original columns
    bool_columns = ['default', 'housing', 'loan']
    for bool_col in bool_columns:
        cleaned_df[bool_col + '_bool'] = df.apply(lambda row: get_dummy_from_bool(row, bool_col), axis=1)

    cleaned_df = cleaned_df.drop(columns=bool_columns)

    # convert categorical columns to dummies

    # drop irrelevant columns
    cleaned_df = cleaned_df.drop(columns=['pdays'])
    #pdb.set_trace()
    # impute incorrect values and drop original columns
    cleaned_df['campaign_cleaned'] = df.apply(lambda row: get_correct_values(row, 'campaign', 34, cleaned_df), axis=1)
    cleaned_df['previous_cleaned'] = df.apply(lambda row: get_correct_values(row, 'previous', 34, cleaned_df), axis=1)

    cleaned_df = cleaned_df.drop(columns=['campaign', 'previous'])

    # ['job', 'marital', 'education', 'contact', 'month', 'poutcome']

    # cat_columns_1 = ['marital', 'month']
    # # for col in cat_columns_1:
    # #     cleaned_df = pd.concat([cleaned_df.drop(col, axis=1),
    # #                             pd.get_dummies(cleaned_df[col], prefix=col, prefix_sep='_',
    # #                                            drop_first=True, dummy_na=False)], axis=1)
    # for col in cat_columns_1:
    #     cleaned_df = pd.concat([cleaned_df.drop(col, axis=1),
    #                             pd.get_dummies(cleaned_df[col], prefix=col, prefix_sep='_',
    #                                            drop_first=True, dummy_na=True)], axis=1)

    cat_columns = ['job', 'education', 'contact','poutcome'] # marital
    cat_columns = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']
    # cleaned_df = trans_num_attrs(cleaned_df, 'age')

    from process import preprocess_data

    # #pdb.set_trace()
    #
    # cleaned_df = encode_edu_attrs(cleaned_df)
    # cleaned_df = encode_job_attrs(cleaned_df)
    # #cleaned_df = encode_marital_attrs(cleaned_df)
    # cleaned_df = encode_contact_attrs(cleaned_df)
    # cleaned_df = encode_poutcome_attrs(cleaned_df)

    #pdb.set_trace()

    # # 通过模型预测　unknown
    # for i in cat_columns:  # ['poutcome', 'education', 'contact']
    #     test_data = cleaned_df[cleaned_df[i] == 0]
    #     testX = test_data.drop(cat_columns, axis=1)
    #
    #     #pdb.set_trace()
    #     #test_data[i] = preprocess_data.train_predict_unknown(trainX, trainY, testX, i)
    #     forest = RandomForestClassifier(n_estimators=100)
    #
    #     #pdb.set_trace()
    #     if config.train_flag:
    #         train_data = cleaned_df[cleaned_df[i] != 0]
    #         trainY = train_data[i]
    #         trainX = train_data.drop(cat_columns, axis=1)
    #         forest = forest.fit(trainX, trainY)
    #
    #         comm.save_file(forest, os.path.join(config.resource_path, i + ".features" + ".pkl"))
    #         test_predictY = forest.predict(testX).astype(int)
    #     else:
    #         forest = comm.load_file(os.path.join(config.resource_path, i + ".features" + ".pkl"))
    #         test_predictY = forest.predict(testX).astype(int)
    #
    #
    #     test_data[i] = pd.DataFrame(test_predictY, index=testX.index)
    #     #pdb.set_trace()
    #     cleaned_df = pd.concat([train_data, test_data])


    ############################################
    for col in cat_columns:
        cleaned_df = pd.concat([cleaned_df.drop(col, axis=1),
                                pd.get_dummies(cleaned_df[col], prefix=col, prefix_sep='_',
                                               drop_first=False, dummy_na=False)], axis=1)

    # for col in cat_columns:
    #     cleaned_df = pd.concat([cleaned_df.drop(col, axis=1),
    #                             pd.get_dummies(cleaned_df[col], prefix=col, prefix_sep='_',
    #                                            drop_first=True, dummy_na=True)], axis=1)
    #pdb.set_trace()
    return cleaned_df


def collect_features():
    if config.train_flag:
        datas = data.load_train_data()
    else:
        datas = data.load_test_data()
    features = collect_feature(datas)
    feature_df = features

    return feature_df


if __name__ == "__main__":

    feature_df = collect_features()
    if config.train_flag:
        comm.save_file(feature_df, config.train_feature_path)
    else:
        comm.save_file(feature_df, config.test_feature_path)
