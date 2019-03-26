# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 22:17:20 2017

@author: liulo
"""

from datetime import datetime
import pandas as pd
import config
from process import comm
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn import preprocessing
import pdb

def feature_scaling(data, numeric_attrs):
    for i in numeric_attrs:
        std = data[i].std()
        if std != 0:
            data[i] = (data[i]-data[i].mean()) / std
        else:
            data = data.drop(i, axis=1)
    return data
    

def encode_cate_attrs(data, cate_attrs):
    #data = encode_edu_attrs(data)
    cate_attrs.remove('education')
    for i in cate_attrs:
        dummies_df = pd.get_dummies(data[i])
        dummies_df = dummies_df.rename(columns=lambda x: i+'_'+str(x))
        data = pd.concat([data,dummies_df],axis=1)
        data = data.drop(i, axis=1)
    return data
    

def encode_bin_attrs(data, bin_attrs):    
    for i in bin_attrs:
        data.loc[data[i] == 'no', i] = 0
        data.loc[data[i] == 'yes', i] = 1
    return data



def fill_unknown(data, bin_attrs, cate_attrs, numeric_attrs):
    # fill_attrs = ['education', 'default', 'housing', 'loan']
    fill_attrs = []
    for i in bin_attrs+cate_attrs:
        if data[data[i] == 'unknown']['y'].count() < 500:
            # delete col containing unknown
            data = data[data[i] != 'unknown'] 
        else:
            fill_attrs.append(i)
    pdb.set_trace()

    data = encode_cate_attrs(data, cate_attrs)
    data = encode_bin_attrs(data, bin_attrs)

    pdb.set_trace()
    #data = trans_num_attrs(data, numeric_attrs)
    pdb.set_trace()
    #data['y'] = data['y'].map({'no': 0, 'yes': 1}).astype(int)

    for i in fill_attrs:     # ['poutcome', 'education', 'contact']
        test_data = data[data[i] == 'unknown']
        testX = test_data.drop(fill_attrs, axis=1)
        train_data = data[data[i] != 'unknown']        
        trainY = train_data[i]
        trainX = train_data.drop(fill_attrs, axis=1)    
        test_data[i] = train_predict_unknown(trainX, trainY, testX)
        data = pd.concat([train_data, test_data])


    return data

def train_predict_unknown(trainX, trainY, testX, i):
    forest = RandomForestClassifier(n_estimators=100)

    pdb.set_trace()
    if config.train_flag:
        forest = forest.fit(trainX, trainY)

        comm.save_file(forest, os.path.join(config.resource_path, i+".features" +  ".pkl"))
    else:
        forest = comm.load_file(os.path.join(config.resource_path, i+ ".features" +  ".pkl"))

    test_predictY = forest.predict(testX).astype(int)
    return pd.DataFrame(test_predictY,index=testX.index)
    
    
def preprocess_data(features):
    # input_data_path = "../data/bank-additional/bank-additional-full.csv"
    # processed_data_path = '../processed_data/bank-additional-full.csv'
    # print("Loading data...")
    #data = pd.read_csv(input_data_path, sep=';')
    print("Preprocessing data...")
    numeric_attrs = ['age', 'duration', 'campaign', 'pdays', 'previous',
                     ]
    bin_attrs = ['default', 'housing', 'loan']
    cate_attrs = [  'job', 'marital', 'education','poutcome', 'contact'
                 ] # 'poutcome', ,  'contact'
    # ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month',
    #  'poutcome']
    #data = shuffle(data)
    data = fill_unknown(features, bin_attrs, cate_attrs, numeric_attrs)
    #data.to_csv(processed_data_path, index=False)
    return data


if __name__ == "__main__":
    start_time = datetime.now()
    preprocess_data()
    end_time = datetime.now()
    delta_seconds = (end_time - start_time).seconds
    print("Cost time: {}s".format(delta_seconds))




